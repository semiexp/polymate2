mod fast;
mod naive;

use crate::shape::{dedup_shapes, Answer, Shape};

#[derive(Clone, Copy)]
pub enum SolverKind {
    Auto,
    Naive,
    Fast,
}

#[derive(Clone, Copy)]
pub struct Config {
    pub identify_transformed_answers: bool,
    pub identify_mirrored_answers: bool,
    pub solver: SolverKind,
}

struct DeduplicatedProblem {
    pieces: Vec<Shape>,
    piece_count: Vec<u32>,
    board: Shape,
}

trait RawAnswers {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Answer;
}

fn dedup_pieces(
    pieces: &[Shape],
    piece_count: &[u32],
) -> (Vec<Shape>, Vec<u32>, Vec<Vec<(usize, usize)>>) {
    assert_eq!(pieces.len(), piece_count.len());

    let dedup_map = dedup_shapes(pieces);

    let mut pieces_after_dedup = vec![];
    let mut piece_count_after_dedup = vec![];
    let mut original_piece_ids = vec![];

    for orig_ids in dedup_map {
        pieces_after_dedup.push(pieces[orig_ids[0]].clone());

        let mut ids = vec![];
        for id in orig_ids {
            for i in 0..piece_count[id] {
                ids.push((id, i as usize));
            }
        }

        piece_count_after_dedup.push(ids.len() as u32);
        original_piece_ids.push(ids);
    }

    (
        pieces_after_dedup,
        piece_count_after_dedup,
        original_piece_ids,
    )
}

pub struct Answers {
    raw_answers: Box<dyn RawAnswers>,
    original_piece_ids: Vec<Vec<(usize, usize)>>,
}

impl Answers {
    pub fn len(&self) -> usize {
        self.raw_answers.len()
    }

    pub fn get(&self, index: usize) -> Answer {
        let raw_answer = self.raw_answers.get(index);
        let answer_data = raw_answer
            .data
            .iter()
            .map(|p| p.map(|(a, b)| self.original_piece_ids[a][b]))
            .collect::<Vec<_>>();
        Answer::new(answer_data, raw_answer.dims())
    }
}

pub fn solve(pieces: &[Shape], piece_count: &[u32], board: &Shape, config: Config) -> Answers {
    if config.identify_mirrored_answers {
        assert!(config.identify_transformed_answers);
    }

    let (pieces, piece_count, original_piece_ids) = dedup_pieces(pieces, piece_count);

    let problem = DeduplicatedProblem {
        pieces,
        piece_count,
        board: board.clone(),
    };

    let raw_answers: Box<dyn RawAnswers>;
    match config.solver {
        SolverKind::Auto | SolverKind::Naive => {
            raw_answers = Box::new(naive::solve(&problem, config));
        }
        SolverKind::Fast => {
            raw_answers = Box::new(fast::solve(&problem, config));
        }
    }

    Answers {
        raw_answers,
        original_piece_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Coord;
    use crate::utils::tests::{shape_from_string, shapes_from_strings};

    #[test]
    fn test_solve_pentomino_small() {
        let pentominoes = shapes_from_strings(&[
            // P
            "###
             ##.",
            // U
            "###
             #.#",
            // F
            ".##
             ##.
             .#.",
        ]);

        for solver in [SolverKind::Naive, SolverKind::Fast] {
            let board = Shape::new(vec![true; 15], Coord(1, 5, 3));

            let config = Config {
                identify_transformed_answers: false,
                identify_mirrored_answers: false,
                solver,
            };

            let answers = solve(&pentominoes, &[1; 3], &board, config);
            assert_eq!(answers.len(), 4);
        }
    }

    #[test]
    fn test_solve_irregular_shape() {
        let shapes = shapes_from_strings(&[
            // l
            "###
             #..",
            // U
            "###
             #.#",
        ]);

        let board = shape_from_string(
            ".###
             ####
             #.#.",
        );

        for solver in [SolverKind::Naive, SolverKind::Fast] {
            let config = Config {
                identify_transformed_answers: false,
                identify_mirrored_answers: false,
                solver,
            };

            let answers = solve(&shapes, &[1; 2], &board, config);
            assert_eq!(answers.len(), 1);

            let answer = answers.get(0);
            assert_eq!(answer[(0, 0, 0)], None);
            assert_eq!(answer[(0, 0, 2)], Some((0, 0)));
            assert_eq!(answer[(0, 1, 0)], Some((1, 0)));
        }
    }

    #[test]
    fn test_solve_multiple_pieces() {
        let shapes = shapes_from_strings(&[
            // P
            "###
             ##.",
        ]);

        let board = shape_from_string(
            "####
             ####
             ..##",
        );

        for solver in [SolverKind::Naive, SolverKind::Fast] {
            let config = Config {
                identify_transformed_answers: false,
                identify_mirrored_answers: false,
                solver,
            };

            let answers = solve(&shapes, &[2], &board, config);
            assert_eq!(answers.len(), 1);

            let answer = answers.get(0);
            assert_eq!(answer[(0, 0, 0)], Some((0, 0)));
            assert_eq!(answer[(0, 0, 3)], Some((0, 1)));
            assert_eq!(answer[(0, 1, 1)], Some((0, 0)));
            assert_eq!(answer[(0, 1, 2)], Some((0, 1)));
        }
    }

    #[test]
    fn test_solve_duplicated_pieces() {
        let shapes = shapes_from_strings(&[
            // P
            "###
             ##.",
            // P (duplicated)
            "#.
             ##
             ##",
        ]);

        let board = shape_from_string(
            "####
             ####
             ..##",
        );

        for solver in [SolverKind::Naive, SolverKind::Fast] {
            let config = Config {
                identify_transformed_answers: false,
                identify_mirrored_answers: false,
                solver,
            };

            let answers = solve(&shapes, &[1, 1], &board, config);
            assert_eq!(answers.len(), 1);

            let answer = answers.get(0);
            assert_eq!(answer[(0, 0, 0)], Some((0, 0)));
            assert_eq!(answer[(0, 0, 3)], Some((1, 0)));
            assert_eq!(answer[(0, 1, 1)], Some((0, 0)));
            assert_eq!(answer[(0, 1, 2)], Some((1, 0)));
        }
    }

    #[test]
    #[ignore]
    fn test_solve_pentomino() {
        let pentominoes = shapes_from_strings(&[
            // P
            "###
             ##.",
            // U
            "###
             #.#",
            // L
            "####
             #...",
            // Y
            "####
             .#..",
            // N
            ".###
             ##..",
            // I
            "#####",
            // V
            "###
             #..
             #..",
            // W
            ".##
             ##.
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

        let board = Shape::new(vec![true; 60], Coord(1, 10, 6));

        for solver in [SolverKind::Naive, SolverKind::Fast] {
            let config = Config {
                identify_transformed_answers: false,
                identify_mirrored_answers: false,
                solver,
            };

            let answers = solve(&pentominoes, &[1; 12], &board, config);
            assert_eq!(answers.len(), 2339 * 4);
        }

        for solver in [SolverKind::Naive, SolverKind::Fast] {
            let config = Config {
                identify_transformed_answers: true,
                identify_mirrored_answers: false,
                solver,
            };

            let answers = solve(&pentominoes, &[1; 12], &board, config);
            assert_eq!(answers.len(), 2339);
        }
    }

    #[test]
    fn test_soma_cube() {
        let pieces = shapes_from_strings(&[
            // l
            "###
             #..",
            // t
            "###
             .#.",
            // n
            "##.
             .##",
            // b
            "##
             #.",
            // x
            "#. ##
             .. #.",
            // y
            ".. ##
             #. #.",
            // z
            ".# ##
             .. #.",
        ]);

        let board = Shape::new(vec![true; 27], Coord(3, 3, 3));

        for solver in [SolverKind::Naive, SolverKind::Fast] {
            let config = Config {
                identify_transformed_answers: false,
                identify_mirrored_answers: false,
                solver,
            };

            let answers = solve(&pieces, &[1; 7], &board, config);
            assert_eq!(answers.len(), 240 * 2 * 24);
        }

        for solver in [SolverKind::Naive, SolverKind::Fast] {
            let config = Config {
                identify_transformed_answers: true,
                identify_mirrored_answers: false,
                solver,
            };

            let answers = solve(&pieces, &[1; 7], &board, config);
            assert_eq!(answers.len(), 240 * 2);
        }

        for solver in [SolverKind::Naive, SolverKind::Fast] {
            let config = Config {
                identify_transformed_answers: true,
                identify_mirrored_answers: true,
                solver,
            };

            let answers = solve(&pieces, &[1; 7], &board, config);
            assert_eq!(answers.len(), 240);
        }
    }
}
