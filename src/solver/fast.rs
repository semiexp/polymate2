use super::{Config, DeduplicatedProblem, RawAnswers};
use crate::shape::{transform_bbox, Answer, Coord, CubicGrid, Shape, Transform, TRANSFORMS};
use crate::utils;

struct PieceTransforms {
    variants: Vec<Shape>,
    origins: Vec<Coord>,
    variant_transforms: Vec<Vec<Transform>>,
}

impl PieceTransforms {
    fn new(shape: &Shape) -> PieceTransforms {
        let mut dup_variants = vec![];
        for &transform in &TRANSFORMS {
            dup_variants.push((shape.apply_transform(transform), transform));
        }
        dup_variants.sort();

        let mut variants = vec![];
        let mut variant_transforms = vec![];

        let mut i = 0;
        while i < dup_variants.len() {
            let mut j = i;
            while j < dup_variants.len() && dup_variants[j].0 == dup_variants[i].0 {
                j += 1;
            }

            variants.push(dup_variants[i].0.clone());
            variant_transforms.push(dup_variants[i..j].iter().map(|&(_, t)| t).collect());
            i = j;
        }

        let origins = variants.iter().map(|v| v.origin()).collect::<Vec<_>>();

        PieceTransforms {
            variants,
            origins,
            variant_transforms,
        }
    }

    fn num_variants(&self) -> usize {
        self.variants.len()
    }

    fn find_variant(&self, piece: &Shape) -> Option<usize> {
        for i in 0..self.variants.len() {
            if &self.variants[i] == piece {
                return Some(i);
            }
        }
        None
    }

    fn transform(&self, variant_id: usize, transform: Transform) -> usize {
        let t = transform * self.variant_transforms[variant_id][0];
        for i in 0..self.variant_transforms.len() {
            if self.variant_transforms[i].iter().any(|&x| x == t) {
                return i;
            }
        }
        panic!();
    }

    fn transform_with_origin(
        &self,
        variant_id: usize,
        origin: Coord,
        board_dims: Coord,
        transform: Transform,
    ) -> (usize, Coord) {
        let variant_after_transform = self.transform(variant_id, transform);

        let top = origin - self.origins[variant_id];
        let top_after_transform =
            transform_bbox(top, self.variants[variant_id].dims(), board_dims, transform);

        let new_origin = top_after_transform + self.origins[variant_after_transform];

        (variant_after_transform, new_origin)
    }
}

struct PlacementBitSets {
    data: Vec<u64>,
    size_per_variant: usize,
    num_pieces: usize,
    num_variants: Vec<usize>,
    offset_by_position_and_piece: Vec<usize>,
}

impl PlacementBitSets {
    fn new(base: &Vec<Vec<Vec<Vec<usize>>>>, num_board_blocks: usize) -> PlacementBitSets {
        let num_pieces = base.len();
        for i in 0..num_pieces {
            assert_eq!(base[i].len(), num_board_blocks);
        }

        let size_per_variant = (num_board_blocks + 63) / 64;

        let mut data = vec![];
        let mut num_variants = vec![];

        for i in 0..num_board_blocks {
            for j in 0..num_pieces {
                num_variants.push(base[j][i].len());
                for k in 0..base[j][i].len() {
                    let ofs = data.len();
                    for _ in 0..size_per_variant {
                        data.push(0);
                    }

                    for &idx in &base[j][i][k] {
                        data[ofs + idx / 64] |= 1 << (idx % 64);
                    }
                }
            }
        }

        let mut offset_by_position_and_piece = vec![0; num_board_blocks * num_pieces];
        for i in 1..num_board_blocks * num_pieces {
            offset_by_position_and_piece[i] =
                offset_by_position_and_piece[i - 1] + num_variants[i - 1];
        }

        PlacementBitSets {
            data,
            size_per_variant,
            num_pieces,
            num_variants,
            offset_by_position_and_piece,
        }
    }

    fn num_variants(&self, position: usize, piece_id: usize) -> usize {
        self.num_variants[position * self.num_pieces + piece_id]
    }

    fn get_bitset(&self, position: usize, piece_id: usize, variant_id: usize) -> &[u64] {
        let offset = self.offset_by_position_and_piece[position * self.num_pieces + piece_id]
            + variant_id * self.size_per_variant;
        &self.data[offset..(offset + self.size_per_variant)]
    }
}

struct CompiledProblem {
    board_dims: Coord,
    num_board_blocks: usize,
    board_symmetry: Vec<Transform>, // `Transform`s which preserve the board
    mirror_board_symmetry: Vec<Transform>, // `Transform`s which preserve the board and have determinant -1
    mirror_pairs: Vec<Option<usize>>,      // [piece] -> Some(piece) if mirrored piece exists
    piece_count: Vec<u32>,
    max_piece_count: u32,
    cumulative_piece_count: Vec<u32>,
    coords: Vec<Coord>, // contiguous position -> (x, y, z)

    placements: Vec<Vec<Vec<Vec<usize>>>>, // [piece][position][variant][index]
    placement_bitsets: PlacementBitSets,

    // [piece][position][variant][transform] -> (position, variant)
    // where `transform` is the index in `board_symmetry` or `mirror_board_symmetry`
    placement_transforms: Vec<Vec<Vec<Vec<(usize, usize)>>>>,
    placement_mirror_transforms: Vec<Vec<Vec<Vec<(usize, usize)>>>>,

    config: Config,
}

impl CompiledProblem {
    fn reconstruct_answer(&self, answer: &[(usize, usize)]) -> Answer {
        let mut ret = Answer::new(
            vec![None; self.board_dims.volume() as usize],
            self.board_dims,
        );

        for i in 0..self.piece_count.len() {
            for j in 0..self.piece_count[i] {
                let (pos, variant) = answer[self.cumulative_piece_count[i] as usize + j as usize];
                if pos == !0 {
                    continue;
                }

                let variant = &self.placements[i][pos][variant];
                for &p in variant {
                    ret[self.coords[p]] = Some((i, j as usize));
                }
            }
        }

        ret
    }

    fn is_normal_form(&self, answer: &[(usize, usize)]) -> bool {
        if self.config.identify_transformed_answers {
            let mut buf = vec![(!0, !0); self.max_piece_count as usize];
            let mut buf2 = vec![(!0, !0); self.max_piece_count as usize];
            'outer: for tr in 0..self.board_symmetry.len() {
                for i in 0..self.piece_count.len() {
                    let ofs = self.cumulative_piece_count[i] as usize;
                    for j in 0..(self.piece_count[i] as usize) {
                        if answer[ofs + j] == (!0, !0) {
                            buf[j] = (!0, !0);
                        } else {
                            buf[j] = self.placement_transforms[i][answer[ofs + j].0]
                                [answer[ofs + j].1][tr];
                        }
                    }

                    for j in 0..(self.piece_count[i] as usize) {
                        buf2[j] = answer[ofs + j];
                    }

                    buf.sort();
                    buf2.sort();

                    for j in 0..(self.piece_count[i] as usize) {
                        match buf2[j].cmp(&buf[j]) {
                            std::cmp::Ordering::Less => continue 'outer,
                            std::cmp::Ordering::Greater => return false,
                            std::cmp::Ordering::Equal => {}
                        }
                    }
                }
            }

            if !self.config.identify_mirrored_answers {
                return true;
            }

            let mut is_mirrorable = true;
            for p in 0..self.piece_count.len() {
                let mut num_used_pieces = 0;
                for j in 0..(self.piece_count[p] as usize) {
                    if answer[self.cumulative_piece_count[p] as usize + j] != (!0, !0) {
                        num_used_pieces += 1;
                    }
                }

                if num_used_pieces > 0 && self.mirror_pairs[p].is_none() {
                    is_mirrorable = false;
                    break;
                }

                let q = self.mirror_pairs[p].unwrap();
                if num_used_pieces > self.piece_count[q] as usize {
                    is_mirrorable = false;
                    break;
                }
            }

            if !is_mirrorable {
                return true;
            }

            'outer: for tr in 0..self.mirror_board_symmetry.len() {
                for p in 0..self.piece_count.len() {
                    let ofs_p = self.cumulative_piece_count[p] as usize;

                    let mut num_used_pieces = 0;
                    for j in 0..(self.piece_count[p] as usize) {
                        if answer[ofs_p + j] != (!0, !0) {
                            num_used_pieces += 1;
                        }
                    }

                    if num_used_pieces > 0 && self.mirror_pairs[p].is_none() {
                        continue 'outer;
                    }

                    let q = self.mirror_pairs[p].unwrap();
                    let ofs_q = self.cumulative_piece_count[q] as usize;

                    let mut mirror_num_used_pieces = 0;
                    for j in 0..(self.piece_count[q] as usize) {
                        if answer[ofs_q as usize + j] != (!0, !0) {
                            mirror_num_used_pieces += 1;
                        }
                    }

                    if num_used_pieces > self.piece_count[q] as usize {
                        continue 'outer;
                    }
                    if mirror_num_used_pieces > self.piece_count[p] as usize {
                        continue 'outer;
                    }

                    if num_used_pieces < mirror_num_used_pieces {
                        continue 'outer;
                    }
                    if num_used_pieces > mirror_num_used_pieces {
                        return false;
                    }

                    for j in 0..mirror_num_used_pieces {
                        let (pos, variant) = answer[ofs_q + j];
                        buf[j] = self.placement_mirror_transforms[q][pos][variant][tr];
                    }
                    for j in mirror_num_used_pieces..self.piece_count[p] as usize {
                        buf[j] = (!0, !0);
                    }
                    buf.sort();

                    for j in 0..(self.piece_count[p].min(self.piece_count[q]) as usize) {
                        match answer[ofs_p + j].cmp(&buf[j]) {
                            std::cmp::Ordering::Less => continue 'outer,
                            std::cmp::Ordering::Greater => return false,
                            std::cmp::Ordering::Equal => {}
                        }
                    }
                }
            }
        }

        true
    }
}

fn iter_map<A, F, U>(it: A, f: F) -> Vec<U>
where
    A: Iterator,
    F: Fn(A::Item) -> U,
{
    it.map(f).collect()
}

fn compile(
    pieces: &[Shape],
    piece_count: &[u32],
    board: &Shape,
    config: Config,
) -> CompiledProblem {
    assert_eq!(pieces.len(), piece_count.len());

    let board_symmetry = board.compute_symmetry();
    let mirror_board_symmetry = board.compute_mirroring_symmetry();
    let coords = utils::coord_iterator(board).collect::<Vec<_>>();

    let mut pos_to_idx = CubicGrid::new(vec![None; board.dims().volume() as usize], board.dims());
    for (idx, p) in coords.iter().enumerate() {
        pos_to_idx[*p] = Some(idx);
    }

    let mut placements: Vec<Vec<Vec<Vec<usize>>>> = vec![];
    let mut placement_transforms: Vec<Vec<Vec<Vec<(usize, usize)>>>> = vec![];
    let mut placement_mirror_transforms: Vec<Vec<Vec<Vec<(usize, usize)>>>> = vec![];

    let all_piece_transforms = pieces.iter().map(PieceTransforms::new).collect::<Vec<_>>();
    let all_valid_placements = iter_map(0..pieces.len(), |p| {
        let piece_transforms = &all_piece_transforms[p];

        iter_map(0..coords.len(), |i| {
            let mut placements = vec![];

            let bc = coords[i];

            'outer: for j in 0..piece_transforms.num_variants() {
                let origin = piece_transforms.origins[j];
                let shape_dims = piece_transforms.variants[j].dims();

                if !(bc.ge(origin)) {
                    continue;
                }
                if !(board.dims().ge(bc + shape_dims - origin)) {
                    continue;
                }

                for pc in utils::coord_iterator(&piece_transforms.variants[j]) {
                    if !board[bc + pc - origin] {
                        continue 'outer;
                    }
                }

                placements.push(j);
            }

            placements
        })
    });

    for p in 0..pieces.len() {
        let piece_transforms = &all_piece_transforms[p];
        let valid_placements = &all_valid_placements[p];

        placements.push(iter_map(0..coords.len(), |i| {
            iter_map(0..valid_placements[i].len(), |j| {
                let coord = coords[i];
                let variant = valid_placements[i][j];
                let orig = piece_transforms.origins[variant];
                iter_map(
                    utils::coord_iterator(&piece_transforms.variants[variant]),
                    |p| {
                        let p = p + coord - orig;
                        let idx = pos_to_idx[p].unwrap();
                        idx
                    },
                )
            })
        }));

        placement_transforms.push(iter_map(0..coords.len(), |i| {
            iter_map(0..valid_placements[i].len(), |j| {
                let variant = valid_placements[i][j];

                iter_map(0..board_symmetry.len(), |k| {
                    let (new_variant, new_origin) = piece_transforms.transform_with_origin(
                        variant,
                        coords[i],
                        board.dims(),
                        board_symmetry[k],
                    );

                    let i2 = pos_to_idx[new_origin].unwrap();
                    let j2 = valid_placements[i2].binary_search(&new_variant).unwrap();

                    (i2, j2)
                })
            })
        }));
    }

    let mut mirror_pairs = vec![];
    for p in 0..pieces.len() {
        let piece_transforms = &all_piece_transforms[p];
        let valid_placements = &all_valid_placements[p];

        if mirror_board_symmetry.is_empty() {
            continue;
        }

        let mirror_transform = mirror_board_symmetry[0];

        let mirrored = pieces[p].apply_transform(mirror_transform);
        let mut mirror_pair = None;
        for q in 0..pieces.len() {
            if let Some(variant) = all_piece_transforms[q].find_variant(&mirrored) {
                assert!(mirror_pair.is_none());
                mirror_pair = Some((q, variant));
            }
        }

        if let Some((q, qv)) = mirror_pair {
            mirror_pairs.push(Some(q));

            // mt: mirror_transform
            // mt(piece[p]) == qv(piece[q])
            placement_mirror_transforms.push(iter_map(0..coords.len(), |i| {
                iter_map(0..valid_placements[i].len(), |j| {
                    iter_map(0..mirror_board_symmetry.len(), |k| {
                        let variant = valid_placements[i][j];

                        let ct = mirror_board_symmetry[k]
                            * piece_transforms.variant_transforms[variant][0];
                        //    ct(piece[p])
                        // == (ct * !mt * mt)(piece[p])
                        // == (ct * !mt)(qvt(piece[q]))
                        let t = all_piece_transforms[q].transform(qv, ct * !mirror_transform);

                        let top = coords[i] - piece_transforms.origins[variant];
                        let top_after_transform = transform_bbox(
                            top,
                            piece_transforms.variants[variant].dims(),
                            board.dims(),
                            mirror_board_symmetry[k],
                        );
                        let new_origin = top_after_transform + all_piece_transforms[q].origins[t];

                        let i2 = pos_to_idx[new_origin].unwrap();
                        let j2 = all_valid_placements[q][i2].binary_search(&t).unwrap();

                        (i2, j2)
                    })
                })
            }));
        } else {
            placement_mirror_transforms.push(vec![]);
        }
    }

    let mut cumulative_piece_count = vec![0; piece_count.len() + 1];
    {
        let mut c = 0;
        for i in 0..piece_count.len() {
            c += piece_count[i];
            cumulative_piece_count[i + 1] = c;
        }
    }

    let placement_bitsets = PlacementBitSets::new(&placements, coords.len());

    CompiledProblem {
        board_dims: board.dims(),
        num_board_blocks: coords.len(),
        board_symmetry,
        mirror_board_symmetry,
        mirror_pairs,
        piece_count: piece_count.to_vec(),
        max_piece_count: *piece_count.iter().max().unwrap(),
        cumulative_piece_count,
        coords,
        placements,
        placement_bitsets,
        placement_transforms,
        placement_mirror_transforms,
        config,
    }
}

fn search(
    piece_count: &mut [u32],
    board: &mut [u64],
    current_answer: &mut [(usize, usize)],
    answers: &mut Vec<Vec<(usize, usize)>>,
    pos: usize,
    problem: &CompiledProblem,
) {
    let mut pos = pos;
    while pos < problem.num_board_blocks && (board[pos / 64] & (1 << (pos % 64))) != 0 {
        pos += 1;
    }

    if pos == problem.num_board_blocks {
        let mut answer = current_answer.to_vec();
        for i in 0..problem.piece_count.len() {
            let start = problem.cumulative_piece_count[i] as usize;
            let end = problem.cumulative_piece_count[i + 1] as usize;
            answer[start..end].reverse();
        }

        if !problem.is_normal_form(&answer) {
            return;
        }

        answers.push(answer);
        return;
    }

    for i in 0..piece_count.len() {
        if piece_count[i] == 0 {
            continue;
        }

        'outer: for j in 0..problem.placement_bitsets.num_variants(pos, i) {
            let mask = problem.placement_bitsets.get_bitset(pos, i, j);
            for k in 0..problem.placement_bitsets.size_per_variant {
                if (mask[k] & board[k]) != 0 {
                    continue 'outer;
                }
            }

            piece_count[i] -= 1;
            current_answer[(piece_count[i] + problem.cumulative_piece_count[i]) as usize] =
                (pos, j);
            for k in 0..problem.placement_bitsets.size_per_variant {
                board[k] ^= mask[k];
            }

            search(
                piece_count,
                board,
                current_answer,
                answers,
                pos + 1,
                problem,
            );

            for k in 0..problem.placement_bitsets.size_per_variant {
                board[k] ^= mask[k];
            }
            current_answer[(piece_count[i] + problem.cumulative_piece_count[i]) as usize] =
                (!0, !0);
            piece_count[i] += 1;
        }
    }
}

struct RawAnswersImpl {
    answers_base: Vec<Vec<(usize, usize)>>,
    compiled_problem: CompiledProblem,
}

impl RawAnswers for RawAnswersImpl {
    fn len(&self) -> usize {
        self.answers_base.len()
    }

    fn get(&self, index: usize) -> Answer {
        self.compiled_problem
            .reconstruct_answer(&self.answers_base[index])
    }
}

pub fn solve(problem: &DeduplicatedProblem, config: Config) -> impl RawAnswers {
    let compiled_problem = compile(
        &problem.pieces,
        &problem.piece_count,
        &problem.board,
        config,
    );

    let mut piece_count = compiled_problem.piece_count.clone();

    let mut board = vec![0; compiled_problem.placement_bitsets.size_per_variant];
    let mut current_answer =
        vec![(!0, !0); *compiled_problem.cumulative_piece_count.last().unwrap() as usize];
    let mut answers = vec![];

    let mut use_pre_pruning = false;
    if config.identify_transformed_answers {
        let mut num_all_blocks = 0;
        for i in 0..piece_count.len() {
            num_all_blocks +=
                piece_count[i] as usize * utils::coord_iterator(&problem.pieces[i]).count();
        }

        if piece_count[0] == 1 && num_all_blocks == compiled_problem.num_board_blocks {
            use_pre_pruning = true;
        }
    }

    if use_pre_pruning {
        let mut always_mirrorable = true;
        for p in 0..piece_count.len() {
            if let Some(q) = compiled_problem.mirror_pairs[p] {
                if piece_count[p] != piece_count[q] {
                    always_mirrorable = false;
                    break;
                }
            } else {
                always_mirrorable = false;
                break;
            }
        }

        // pre-pruning
        let piece = 0;
        for pos in 0..compiled_problem.num_board_blocks {
            for i in 0..compiled_problem.placements[piece][pos].len() {
                let mut is_pruned = false;
                for j in 0..compiled_problem.board_symmetry.len() {
                    if (pos, i) > compiled_problem.placement_transforms[piece][pos][i][j] {
                        is_pruned = true;
                        break;
                    }
                }
                if config.identify_mirrored_answers
                    && compiled_problem.mirror_pairs[piece] == Some(piece)
                    && always_mirrorable
                {
                    for j in 0..compiled_problem.mirror_board_symmetry.len() {
                        if (pos, i) > compiled_problem.placement_mirror_transforms[piece][pos][i][j]
                        {
                            is_pruned = true;
                            break;
                        }
                    }
                }
                if is_pruned {
                    continue;
                }

                let mask = compiled_problem.placement_bitsets.get_bitset(pos, piece, i);
                for k in 0..compiled_problem.placement_bitsets.size_per_variant {
                    if (mask[k] & board[k]) != 0 {
                        panic!();
                    }
                }

                piece_count[piece] -= 1;
                current_answer[(piece_count[piece]
                    + compiled_problem.cumulative_piece_count[piece] as u32)
                    as usize] = (pos, i);
                for k in 0..compiled_problem.placement_bitsets.size_per_variant {
                    board[k] ^= mask[k];
                }

                search(
                    &mut piece_count,
                    &mut board,
                    &mut current_answer,
                    &mut answers,
                    0,
                    &compiled_problem,
                );

                for k in 0..compiled_problem.placement_bitsets.size_per_variant {
                    board[k] ^= mask[k];
                }
                current_answer[(piece_count[piece]
                    + compiled_problem.cumulative_piece_count[piece] as u32)
                    as usize] = (!0, !0);
                piece_count[piece] += 1;
            }
        }
    } else {
        search(
            &mut piece_count,
            &mut board,
            &mut current_answer,
            &mut answers,
            0,
            &compiled_problem,
        );
    }

    RawAnswersImpl {
        answers_base: answers,
        compiled_problem,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::get_shapes_by_names;

    #[test]
    fn test_piece_transforms() {
        let (shapes, _) = get_shapes_by_names("moPULIVTZFX");
        let expected_num_variants = [1, 3, 24, 12, 24, 3, 12, 12, 12, 24, 3];

        for (shape, n) in shapes.into_iter().zip(expected_num_variants.into_iter()) {
            let piece = PieceTransforms::new(&shape);
            assert_eq!(piece.num_variants(), n);

            for v in 0..piece.num_variants() {
                for &transform in &TRANSFORMS {
                    let expected = &piece.variants[v].apply_transform(transform);
                    let actual = &piece.variants[piece.transform(v, transform)];
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
