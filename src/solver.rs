// TODO: tentative implementation

use crate::shape::Shape;

struct CompiledProblem {
    board_size: usize,
    piece_count: usize,
    placements: Vec<Vec<Vec<Vec<usize>>>>, // [position][piece][variant][index]
}

fn compile(pieces: &[Shape], board: &Shape) -> CompiledProblem {
    let board_dim = board.dims();

    let mut board_size = 0;
    let mut pos_id = vec![vec![vec![!0; board_dim.2]; board_dim.1]; board_dim.0];
    let mut positions = vec![];
    for i in 0..board_dim.0 {
        for j in 0..board_dim.1 {
            for k in 0..board_dim.2 {
                if board[(i, j, k)] {
                    positions.push((i, j, k));
                    pos_id[i][j][k] = board_size;
                    board_size += 1;
                }
            }
        }
    }

    let piece_variants = pieces
        .iter()
        .map(|piece| piece.enumerate_transforms())
        .collect::<Vec<_>>();

    let mut placements = vec![];
    for (i, j, k) in positions {
        let mut placements_pos = vec![];
        for pv in &piece_variants {
            let mut placements_piece = vec![];

            'outer: for variant in pv {
                let origin = variant.origin();
                let shape_dims = variant.dims();

                if !(i >= origin.0 && j >= origin.1 && k >= origin.2) {
                    continue;
                }
                if !(i + shape_dims.0 - origin.0 <= board_dim.0
                    && j + shape_dims.1 - origin.1 <= board_dim.1
                    && k + shape_dims.2 - origin.2 <= board_dim.2)
                {
                    continue;
                }

                for pi in 0..shape_dims.0 {
                    for pj in 0..shape_dims.1 {
                        for pk in 0..shape_dims.2 {
                            if variant[(pi, pj, pk)] {
                                if !board[(i + pi - origin.0, j + pj - origin.1, k + pk - origin.2)]
                                {
                                    continue 'outer;
                                }
                            }
                        }
                    }
                }

                let mut placement = vec![];
                for pi in 0..shape_dims.0 {
                    for pj in 0..shape_dims.1 {
                        for pk in 0..shape_dims.2 {
                            if variant[(pi, pj, pk)] {
                                let ti = i + pi - origin.0;
                                let tj = j + pj - origin.1;
                                let tk = k + pk - origin.2;

                                let v = pos_id[ti][tj][tk];
                                assert_ne!(v, !0);
                                placement.push(v);
                            }
                        }
                    }
                }
                placements_piece.push(placement);
            }
            placements_pos.push(placements_piece);
        }
        placements.push(placements_pos);
    }

    eprintln!("{:?}", placements[1][0]);

    CompiledProblem {
        board_size,
        piece_count: pieces.len(),
        placements,
    }
}

fn search(
    piece_count: &mut [i32],
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
        answers.push(current_answer.to_vec());
        return;
    }

    for i in 0..problem.piece_count {
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

pub fn solve(pieces: &[Shape], board: &Shape) -> Vec<Vec<(usize, usize)>> {
    let problem = compile(pieces, board);

    let mut piece_count = vec![1; problem.piece_count];

    let mut board = vec![false; problem.board_size];
    let mut current_answer = vec![(!0, !0); problem.board_size];
    let mut answers = vec![];
    search(
        &mut piece_count,
        &mut board,
        &mut current_answer,
        &mut answers,
        0,
        &problem,
    );

    answers
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

        let answers = solve(&pentominoes, &board);
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

        let answers = solve(&shapes, &board);
        assert_eq!(answers.len(), 1);
    }

    #[test]
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

        let board = Shape::new(vec![true; 60], (5, 4, 3));

        let answers = solve(&pentominoes, &board);
        assert_eq!(answers.len(), 3940 * 8);
    }
}
