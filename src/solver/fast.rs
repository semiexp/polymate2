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
    piece_count: Vec<u32>,
    max_piece_count: u32,
    cumulative_piece_count: Vec<u32>,
    coords: Vec<Coord>, // contiguous position -> (x, y, z)

    placements: Vec<Vec<Vec<Vec<usize>>>>, // [piece][position][variant][index]
    placement_bitsets: PlacementBitSets,

    // [piece][position][variant][transform] -> (position, variant)
    // where `transform` is the index in `board_symmetry`
    placement_transforms: Vec<Vec<Vec<Vec<(usize, usize)>>>>,

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

                    buf.sort();

                    for j in 0..(self.piece_count[i] as usize) {
                        match answer[ofs + j].cmp(&buf[j]) {
                            std::cmp::Ordering::Less => continue 'outer,
                            std::cmp::Ordering::Greater => return false,
                            std::cmp::Ordering::Equal => {}
                        }
                    }
                }
            }

            // TODO: support mirrored transforms
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
    let coords = utils::coord_iterator(board).collect::<Vec<_>>();

    let mut pos_to_idx = CubicGrid::new(vec![None; board.dims().volume() as usize], board.dims());
    for (idx, p) in coords.iter().enumerate() {
        pos_to_idx[*p] = Some(idx);
    }

    let mut placements: Vec<Vec<Vec<Vec<usize>>>> = vec![];
    let mut placement_transforms: Vec<Vec<Vec<Vec<(usize, usize)>>>> = vec![];

    for p in 0..pieces.len() {
        let transforms = PieceTransforms::new(&pieces[p]);

        let mut all_placements = vec![]; // [position][i] -> variant_id

        for i in 0..coords.len() {
            let mut placements = vec![];

            let bc = coords[i];

            'outer: for j in 0..transforms.num_variants() {
                let origin = transforms.origins[j];
                let shape_dims = transforms.variants[j].dims();

                if !(bc.ge(origin)) {
                    continue;
                }
                if !(board.dims().ge(bc + shape_dims - origin)) {
                    continue;
                }

                for pc in utils::coord_iterator(&transforms.variants[j]) {
                    if !board[bc + pc - origin] {
                        continue 'outer;
                    }
                }

                placements.push(j);
            }

            all_placements.push(placements);
        }

        placements.push(iter_map(0..coords.len(), |i| {
            iter_map(0..all_placements[i].len(), |j| {
                let coord = coords[i];
                let variant = all_placements[i][j];
                let orig = transforms.origins[variant];
                iter_map(utils::coord_iterator(&transforms.variants[variant]), |p| {
                    let p = p + coord - orig;
                    let idx = pos_to_idx[p].unwrap();
                    idx
                })
            })
        }));

        placement_transforms.push(iter_map(0..coords.len(), |i| {
            iter_map(0..all_placements[i].len(), |j| {
                let variant = all_placements[i][j];

                iter_map(0..board_symmetry.len(), |k| {
                    let (new_variant, new_origin) = transforms.transform_with_origin(
                        variant,
                        coords[i],
                        board.dims(),
                        board_symmetry[k],
                    );

                    let i2 = pos_to_idx[new_origin].unwrap();
                    let j2 = all_placements[i2].binary_search(&new_variant).unwrap();

                    (i2, j2)
                })
            })
        }));
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
        piece_count: piece_count.to_vec(),
        max_piece_count: *piece_count.iter().max().unwrap(),
        cumulative_piece_count,
        coords,
        placements,
        placement_bitsets,
        placement_transforms,
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
    if config.identify_mirrored_answers {
        todo!();
    }

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

    #[test]
    fn test_piece_transforms() {
        let shapes = crate::utils::tests::shapes_from_strings(&[
            // (monocube)
            "#",
            // o
            "##
             ##",
            // P
            "###
             ##.",
            // U
            "###
             #.#",
            // L
            "####
             #...",
            // I
            "#####",
            // V
            "###
             #..
             #..",
            // T
            "###
             .#.
             .#.",
            // Z
            ".##
             .#.
             ##.",
            // F
            ".##
             ##.
             .#.",
            // X
            ".#.
             ###
             .#.",
        ]);
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
