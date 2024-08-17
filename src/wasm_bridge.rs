use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use crate::shape::Answer as InternalAnswer;
use crate::shape::Shape as InternalShape;

#[wasm_bindgen]
pub struct Answers {
    answers: Vec<InternalAnswer>,
}

#[derive(Tsify, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct Answer {
    data: Vec<(i32, i32)>,
}

#[wasm_bindgen]
impl Answers {
    pub fn len(&self) -> usize {
        self.answers.len()
    }

    pub fn get(&self, index: usize) -> Answer {
        let answer = &self.answers[index];
        let data = answer
            .data
            .iter()
            .map(|&p| match p {
                Some((a, b)) => (a as i32, b as i32),
                None => (-1, -1),
            })
            .collect();
        Answer { data }
    }
}

#[derive(Tsify, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct Problem {
    pieces: Vec<Vec<Vec<Vec<i32>>>>,

    #[tsify(optional)]
    piece_count: Option<Vec<u32>>,
    board: Vec<Vec<Vec<i32>>>,
}

fn make_shape(data: Vec<Vec<Vec<i32>>>) -> InternalShape {
    let dims = (data.len(), data[0].len(), data[0][0].len());
    let mut shape_data = vec![];
    for layer in data {
        assert_eq!(layer.len(), dims.1);
        for row in layer {
            assert_eq!(row.len(), dims.2);
            for cell in row {
                shape_data.push(cell == 1);
            }
        }
    }
    InternalShape::new(shape_data, dims)
}

#[wasm_bindgen]
pub fn solve(problem: Problem) -> Answers {
    let pieces = problem
        .pieces
        .iter()
        .map(|data| make_shape(data.clone()))
        .collect::<Vec<_>>();
    let board = make_shape(problem.board);

    let piece_count = problem.piece_count.unwrap_or_else(|| vec![1; pieces.len()]);
    let answers = crate::solver::solve(&pieces, &piece_count, &board);
    Answers { answers }
}
