use crate::shape::{Coord, Shape};

pub fn coord_iterator(shape: &Shape) -> impl Iterator<Item = Coord> {
    let mut positions = vec![];
    let dims = shape.dims();

    for i in 0..dims.0 {
        for j in 0..dims.1 {
            for k in 0..dims.2 {
                let c = Coord(i, j, k);
                if shape[c] {
                    positions.push(c);
                }
            }
        }
    }

    return positions.into_iter();
}
