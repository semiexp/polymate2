use crate::shape::Shape;

pub fn position_iterator(shape: &Shape) -> impl Iterator<Item = (usize, usize, usize)> {
    let mut positions = vec![];
    let dims = shape.dims();

    for i in 0..dims.0 {
        for j in 0..dims.1 {
            for k in 0..dims.2 {
                if shape[(i, j, k)] {
                    positions.push((i, j, k));
                }
            }
        }
    }

    return positions.into_iter();
}
