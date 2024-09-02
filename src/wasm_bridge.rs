use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use crate::shape::Shape;
use crate::solver::{Answers, SolverKind};

#[wasm_bindgen(js_name = Answers)]
pub struct JsAnswers {
    answers: Answers,
}

#[derive(Tsify, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename = "Answer")]
pub struct JsAnswer {
    data: Vec<(i32, i32)>,
}

#[wasm_bindgen(js_class = Answers)]
impl JsAnswers {
    pub fn len(&self) -> usize {
        self.answers.len()
    }

    pub fn get(&self, index: usize) -> JsAnswer {
        let answer = self.answers.get(index);
        let data = answer
            .data
            .iter()
            .map(|&p| match p {
                Some((a, b)) => (a as i32, b as i32),
                None => (-1, -1),
            })
            .collect();
        JsAnswer { data }
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

fn make_shape(data: Vec<Vec<Vec<i32>>>) -> Shape {
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
    Shape::new(shape_data, dims)
}

#[wasm_bindgen]
pub fn solve(problem: Problem) -> JsAnswers {
    let pieces = problem
        .pieces
        .iter()
        .map(|data| make_shape(data.clone()))
        .collect::<Vec<_>>();
    let board = make_shape(problem.board);

    let piece_count = problem.piece_count.unwrap_or_else(|| vec![1; pieces.len()]);
    let config = crate::solver::Config {
        identify_transformed_answers: true,
        identify_mirrored_answers: true,
        solver: SolverKind::Naive,
    };
    let answers = crate::solver::solve(&pieces, &piece_count, &board, config);
    JsAnswers { answers }
}
