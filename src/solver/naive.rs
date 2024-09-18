use std::collections::HashMap;

use super::{Config, DeduplicatedProblem, RawAnswers};
use crate::shape::{Answer, Coord, CubicGrid, Shape, Transform};
use crate::utils;

struct CompiledProblem {
    board_dims: Coord,
    board_size: usize,
    board_symmetry: Vec<Transform>,
    board_mirror_symmetry: Vec<Transform>,
    piece_count: Vec<u32>,
    coords: Vec<Coord>,                    // contiguous position -> (x, y, z)
    placements: Vec<Vec<Vec<Vec<usize>>>>, // [position][piece][variant][index]
    mirrored_piece_ids: Vec<Vec<Option<(usize, usize)>>>,
    config: Config,
}

impl CompiledProblem {
    fn reconstruct_answer(&self, answer: &[(usize, usize)]) -> Answer {
        assert_eq!(answer.len(), self.board_size);

        let mut ret = Answer::new(
            vec![None; self.board_dims.volume() as usize],
            self.board_dims,
        );
        let mut used_pieces = vec![0; self.piece_count.len()];

        for i in 0..self.board_size {
            let (piece, variant) = answer[i];
            if piece == !0 {
                continue;
            }

            let variant = &self.placements[i][piece][variant];
            for &p in variant {
                ret[self.coords[p]] = Some((piece, used_pieces[piece]));
            }
            used_pieces[piece] += 1;
        }

        ret
    }

    fn to_mirrored(&self, answer: &Answer) -> Option<Answer> {
        let mut ret = Answer::new(vec![None; answer.data.len()], answer.dims());

        for i in 0..answer.data.len() {
            if let Some((piece, piece_id)) = answer.data[i] {
                if let Some(e) = self.mirrored_piece_ids[piece][piece_id] {
                    ret.data[i] = Some(e);
                } else {
                    return None;
                }
            }
        }

        Some(ret)
    }

    fn is_normal_form(&self, answer: &Answer) -> bool {
        if self.config.identify_transformed_answers {
            for tr in &self.board_symmetry {
                let mut answer_transformed = answer.apply_transform(*tr);
                renormalize_piece_indices(&mut answer_transformed, self.piece_count.len());
                if !(answer <= &answer_transformed) {
                    return false;
                }
            }
            if self.config.identify_mirrored_answers {
                for tr in &self.board_mirror_symmetry {
                    let mut answer_transformed = answer.apply_transform(*tr);
                    answer_transformed = self.to_mirrored(&answer_transformed).unwrap();
                    renormalize_piece_indices(&mut answer_transformed, self.piece_count.len());
                    if !(answer <= &answer_transformed) {
                        return false;
                    }
                }
            }
        }

        true
    }
}

fn renormalize_piece_indices(answer: &mut Answer, num_pieces: usize) {
    let mut translate_map = HashMap::<(usize, usize), usize>::new();
    let mut cur_idx = vec![0; num_pieces];

    for p in &mut answer.data {
        if let Some((piece, variant)) = p {
            if let Some(&new_idx) = translate_map.get(&(*piece, *variant)) {
                *variant = new_idx;
            } else {
                let new_idx = cur_idx[*piece];
                cur_idx[*piece] += 1;
                translate_map.insert((*piece, *variant), new_idx);
                *variant = new_idx;
            }
        }
    }
}

fn compute_mirrored_piece_ids(
    pieces: &[Shape],
    piece_count: &[u32],
) -> Vec<Vec<Option<(usize, usize)>>> {
    let mut mirrored_piece_ids = vec![];

    let mut normalized_piece_and_mirror = vec![];
    for piece in pieces {
        let normalized_piece = piece.normalize();
        let mirrored_piece = piece.mirror().normalize();
        normalized_piece_and_mirror.push((normalized_piece, mirrored_piece));
    }

    for i in 0..pieces.len() {
        let mut mirrored_piece_id = None;
        for j in 0..pieces.len() {
            if normalized_piece_and_mirror[i].0 == normalized_piece_and_mirror[j].1 {
                assert!(mirrored_piece_id.is_none(), "pieces should be deduplicated");
                mirrored_piece_id = Some(j);
            }
        }

        if let Some(j) = mirrored_piece_id {
            let mut ids = vec![];
            for k in 0..piece_count[i] {
                if k < piece_count[j] {
                    ids.push(Some((j, k as usize)));
                } else {
                    ids.push(None);
                }
            }
            mirrored_piece_ids.push(ids);
        } else {
            mirrored_piece_ids.push(vec![None; piece_count[i] as usize]);
        }
    }

    mirrored_piece_ids
}

fn compile(
    pieces: &[Shape],
    piece_count: &[u32],
    board: &Shape,
    config: Config,
) -> CompiledProblem {
    assert_eq!(pieces.len(), piece_count.len());

    let mirrored_piece_ids = compute_mirrored_piece_ids(&pieces, &piece_count);

    let board_dims = board.dims();

    let coords = utils::coord_iterator(board).collect::<Vec<_>>();
    let board_size = coords.len();
    let mut pos_id = CubicGrid::new(vec![!0; board_dims.volume() as usize], board_dims);
    for (idx, p) in coords.iter().enumerate() {
        pos_id[*p] = idx;
    }

    let piece_variants = pieces
        .iter()
        .map(|piece| piece.enumerate_transforms())
        .collect::<Vec<_>>();

    let mut placements = vec![];
    for &bc in &coords {
        let mut placements_pos: Vec<Vec<Vec<usize>>> = vec![];
        for pv in &piece_variants {
            let mut placements_piece = vec![];

            'outer: for variant in pv {
                let origin = variant.origin();
                let shape_dims = variant.dims();

                if !(bc.ge(origin)) {
                    continue;
                }
                if !(board_dims.ge(bc + shape_dims - origin)) {
                    continue;
                }

                for pc in utils::coord_iterator(variant) {
                    if !board[bc + pc - origin] {
                        continue 'outer;
                    }
                }

                let mut placement = vec![];
                for pc in utils::coord_iterator(variant) {
                    let v = pos_id[bc + pc - origin];
                    assert_ne!(v, !0);
                    placement.push(v);
                }
                placements_piece.push(placement);
            }
            placements_pos.push(placements_piece);
        }
        placements.push(placements_pos);
    }

    CompiledProblem {
        board_dims,
        board_size,
        board_symmetry: board.compute_symmetry(),
        board_mirror_symmetry: board.compute_mirroring_symmetry(),
        piece_count: piece_count.to_vec(),
        coords,
        placements,
        mirrored_piece_ids,
        config,
    }
}

fn search(
    piece_count: &mut [u32],
    board: &mut [bool],
    current_answer: &mut [(usize, usize)],
    answers: &mut Vec<Vec<(usize, usize)>>,
    pos: usize,
    problem: &CompiledProblem,
) {
    let mut pos = pos;
    while pos < problem.board_size && board[pos] {
        pos += 1;
    }

    if pos == problem.board_size {
        let answer = problem.reconstruct_answer(current_answer);
        if !problem.is_normal_form(&answer) {
            return;
        }

        answers.push(current_answer.to_vec());
        return;
    }

    for i in 0..piece_count.len() {
        if piece_count[i] == 0 {
            continue;
        }

        'outer: for (j, variant) in problem.placements[pos][i].iter().enumerate() {
            for p in variant {
                if board[*p] {
                    continue 'outer;
                }
            }

            current_answer[pos] = (i, j);
            piece_count[i] -= 1;
            for p in variant {
                board[*p] = true;
            }

            search(
                piece_count,
                board,
                current_answer,
                answers,
                pos + 1,
                problem,
            );

            for p in variant {
                board[*p] = false;
            }
            current_answer[pos] = (!0, !0);
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

    let mut board = vec![false; compiled_problem.board_size];
    let mut current_answer = vec![(!0, !0); compiled_problem.board_size];
    let mut answers = vec![];
    search(
        &mut piece_count,
        &mut board,
        &mut current_answer,
        &mut answers,
        0,
        &compiled_problem,
    );

    RawAnswersImpl {
        answers_base: answers,
        compiled_problem,
    }
}
