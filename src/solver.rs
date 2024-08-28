// TODO: tentative implementation

use std::collections::HashMap;

use crate::shape::{Answer, CubicGrid, Shape, Transform};
use crate::utils;

#[derive(Clone, Copy)]
pub struct Config {
    pub identify_transformed_answers: bool,
}

struct CompiledProblem {
    board_dims: (usize, usize, usize),
    board_size: usize,
    board_symmetry: Vec<Transform>,
    piece_count: Vec<u32>,
    positions: Vec<(usize, usize, usize)>, // contiguous position -> (x, y, z)
    placements: Vec<Vec<Vec<Vec<usize>>>>, // [position][piece][variant][index]
    config: Config,
}

impl CompiledProblem {
    fn reconstruct_answer(&self, answer: &[(usize, usize)]) -> Answer {
        assert_eq!(answer.len(), self.board_size);

        let volume = self.board_dims.0 * self.board_dims.1 * self.board_dims.2;
        let mut ret = Answer::new(vec![None; volume], self.board_dims);
        let mut used_pieces = vec![0; self.piece_count.len()];

        for i in 0..self.board_size {
            let (piece, variant) = answer[i];
            if piece == !0 {
                continue;
            }

            let variant = &self.placements[i][piece][variant];
            for &p in variant {
                ret[self.positions[p]] = Some((piece, used_pieces[piece]));
            }
            used_pieces[piece] += 1;
        }

        ret
    }
}

fn compile(
    pieces: &[Shape],
    piece_count: Vec<u32>,
    board: &Shape,
    config: Config,
) -> CompiledProblem {
    assert_eq!(pieces.len(), piece_count.len());

    let board_dims = board.dims();

    let mut board_size = 0;
    let mut pos_id = vec![vec![vec![!0; board_dims.2]; board_dims.1]; board_dims.0];
    let mut positions = vec![];
    for (i, j, k) in utils::position_iterator(board) {
        positions.push((i, j, k));
        pos_id[i][j][k] = board_size;
        board_size += 1;
    }

    let piece_variants = pieces
        .iter()
        .map(|piece| piece.enumerate_transforms())
        .collect::<Vec<_>>();

    let mut placements = vec![];
    for &(i, j, k) in &positions {
        let mut placements_pos = vec![];
        for pv in &piece_variants {
            let mut placements_piece = vec![];

            'outer: for variant in pv {
                let origin = variant.origin();
                let shape_dims = variant.dims();

                if !(i >= origin.0 && j >= origin.1 && k >= origin.2) {
                    continue;
                }
                if !(i + shape_dims.0 - origin.0 <= board_dims.0
                    && j + shape_dims.1 - origin.1 <= board_dims.1
                    && k + shape_dims.2 - origin.2 <= board_dims.2)
                {
                    continue;
                }

                for (pi, pj, pk) in utils::position_iterator(variant) {
                    if !board[(i + pi - origin.0, j + pj - origin.1, k + pk - origin.2)] {
                        continue 'outer;
                    }
                }

                let mut placement = vec![];
                for (pi, pj, pk) in utils::position_iterator(variant) {
                    let ti = i + pi - origin.0;
                    let tj = j + pj - origin.1;
                    let tk = k + pk - origin.2;

                    let v = pos_id[ti][tj][tk];
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
        piece_count,
        positions,
        placements,
        config,
    }
}

fn renormalize_piece_indices(answer: &mut CubicGrid<Option<(usize, usize)>>, num_pieces: usize) {
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
        if problem.config.identify_transformed_answers && problem.board_symmetry.len() > 1 {
            let answer = problem.reconstruct_answer(current_answer);
            for tr in &problem.board_symmetry {
                let mut answer_transformed = answer.apply_transform(*tr);
                renormalize_piece_indices(&mut answer_transformed, problem.piece_count.len());
                if !(answer <= answer_transformed) {
                    return;
                }
            }
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

fn solve_raw(compiled_problem: &CompiledProblem) -> Vec<Vec<(usize, usize)>> {
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

    answers
}

pub fn solve(pieces: &[Shape], piece_count: &[u32], board: &Shape, config: Config) -> Vec<Answer> {
    let problem = compile(pieces, piece_count.to_vec(), board, config);

    let answers_raw = solve_raw(&problem);

    answers_raw
        .into_iter()
        .map(|answer| problem.reconstruct_answer(&answer))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_pentomino_small() {
        let pentominoes = vec![
            Shape::from_array_2d(vec![vec![true, true, true], vec![true, true, false]]),
            Shape::from_array_2d(vec![vec![true, true, true], vec![true, false, true]]),
            Shape::from_array_2d(vec![
                vec![false, true, true],
                vec![true, true, false],
                vec![false, true, false],
            ]),
        ];

        let board = Shape::new(vec![true; 60], (1, 5, 3));

        let config = Config {
            identify_transformed_answers: false,
        };

        let answers = solve(&pentominoes, &[1; 3], &board, config);
        assert_eq!(answers.len(), 4);
    }

    #[test]
    fn test_solve_irregular_shape() {
        let shapes = vec![
            Shape::from_array_2d(vec![vec![true, true, true], vec![true, false, false]]),
            Shape::from_array_2d(vec![vec![true, true, true], vec![true, false, true]]),
        ];

        let board = Shape::from_array_2d(vec![
            vec![false, true, true, true],
            vec![true, true, true, true],
            vec![true, false, true, false],
        ]);

        let config = Config {
            identify_transformed_answers: false,
        };

        let answers = solve(&shapes, &[1; 2], &board, config);
        assert_eq!(answers.len(), 1);

        let answer = &answers[0];
        assert_eq!(answer[(0, 0, 0)], None);
        assert_eq!(answer[(0, 0, 2)], Some((0, 0)));
        assert_eq!(answer[(0, 1, 0)], Some((1, 0)));
    }

    #[test]
    fn test_solve_multiple_pieces() {
        let shapes = vec![Shape::from_array_2d(vec![
            vec![true, true, true],
            vec![true, true, false],
        ])];

        let board = Shape::from_array_2d(vec![
            vec![true, true, true, true],
            vec![true, true, true, true],
            vec![false, false, true, true],
        ]);

        let config = Config {
            identify_transformed_answers: false,
        };

        let answers = solve(&shapes, &[2], &board, config);
        assert_eq!(answers.len(), 1);

        let answer = &answers[0];
        assert_eq!(answer[(0, 0, 0)], Some((0, 0)));
        assert_eq!(answer[(0, 0, 3)], Some((0, 1)));
        assert_eq!(answer[(0, 1, 1)], Some((0, 0)));
        assert_eq!(answer[(0, 1, 2)], Some((0, 1)));
    }

    #[test]
    #[ignore]
    fn test_solve_pentomino() {
        let pentominoes = vec![
            Shape::from_array_2d(vec![vec![true, true, true], vec![true, true, false]]),
            Shape::from_array_2d(vec![vec![true, true, true], vec![true, false, true]]),
            Shape::from_array_2d(vec![
                vec![true, true, true, true],
                vec![true, false, false, false],
            ]),
            Shape::from_array_2d(vec![
                vec![true, true, true, true],
                vec![false, true, false, false],
            ]),
            Shape::from_array_2d(vec![
                vec![false, true, true, true],
                vec![true, true, false, false],
            ]),
            Shape::from_array_2d(vec![vec![true, true, true, true, true]]),
            Shape::from_array_2d(vec![
                vec![true, true, true],
                vec![true, false, false],
                vec![true, false, false],
            ]),
            Shape::from_array_2d(vec![
                vec![false, true, true],
                vec![true, true, false],
                vec![true, false, false],
            ]),
            Shape::from_array_2d(vec![
                vec![true, true, true],
                vec![false, true, false],
                vec![false, true, false],
            ]),
            Shape::from_array_2d(vec![
                vec![false, true, true],
                vec![false, true, false],
                vec![true, true, false],
            ]),
            Shape::from_array_2d(vec![
                vec![false, true, true],
                vec![true, true, false],
                vec![false, true, false],
            ]),
            Shape::from_array_2d(vec![
                vec![false, true, false],
                vec![true, true, true],
                vec![false, true, false],
            ]),
        ];

        let board = Shape::new(vec![true; 60], (1, 10, 6));

        {
            let config = Config {
                identify_transformed_answers: false,
            };

            let answers = solve(&pentominoes, &[1; 12], &board, config);
            assert_eq!(answers.len(), 2339 * 4);
        }

        {
            let config = Config {
                identify_transformed_answers: true,
            };

            let answers = solve(&pentominoes, &[1; 12], &board, config);
            assert_eq!(answers.len(), 2339);
        }
    }

    #[test]
    fn test_soma_cube() {
        let pieces = vec![
            Shape::from_array_2d(vec![vec![true, true, true], vec![true, false, false]]),
            Shape::from_array_2d(vec![vec![true, true, true], vec![false, true, false]]),
            Shape::from_array_2d(vec![vec![true, true, false], vec![false, true, true]]),
            Shape::from_array_2d(vec![vec![true, true], vec![true, false]]),
            Shape::from_array_3d(vec![
                vec![vec![true, false], vec![false, false]],
                vec![vec![true, true], vec![true, false]],
            ]),
            Shape::from_array_3d(vec![
                vec![vec![false, true], vec![false, false]],
                vec![vec![true, true], vec![true, false]],
            ]),
            Shape::from_array_3d(vec![
                vec![vec![false, false], vec![true, false]],
                vec![vec![true, true], vec![true, false]],
            ]),
        ];

        let board = Shape::new(vec![true; 27], (3, 3, 3));

        {
            let config = Config {
                identify_transformed_answers: false,
            };

            let answers = solve(&pieces, &[1; 7], &board, config);
            assert_eq!(answers.len(), 240 * 2 * 24);
        }

        {
            let config = Config {
                identify_transformed_answers: true,
            };

            let answers = solve(&pieces, &[1; 7], &board, config);
            assert_eq!(answers.len(), 240 * 2);
        }
    }
}
