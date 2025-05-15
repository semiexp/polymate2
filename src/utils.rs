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

/// A struct to represent the order of axes in a 3D coordinate system.
/// The first element represents the index of the axis which should be most significant.
pub struct AxisOrder(usize, usize, usize);

fn get(coord: &Coord, axis: usize) -> i32 {
    match axis {
        0 => coord.0,
        1 => coord.1,
        2 => coord.2,
        _ => panic!("Invalid axis index"),
    }
}

impl AxisOrder {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        assert!(x != y && x != z && y != z, "Axis indices must be unique");
        assert!(
            x < 3 && y < 3 && z < 3,
            "Axis indices must be in the range [0, 2]"
        );
        AxisOrder(x, y, z)
    }

    fn reorder(&self, coord: &Coord) -> (i32, i32, i32) {
        let x = get(coord, self.0);
        let y = get(coord, self.1);
        let z = get(coord, self.2);
        (x, y, z)
    }

    pub fn compare(&self, a: &Coord, b: &Coord) -> std::cmp::Ordering {
        let a = self.reorder(a);
        let b = self.reorder(b);

        a.cmp(&b)
    }
}

impl Default for AxisOrder {
    fn default() -> Self {
        AxisOrder(0, 1, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axis_order() {
        let order = AxisOrder::new(2, 0, 1);

        assert_eq!(
            order.compare(&Coord(1, 2, 3), &Coord(3, 1, 2)),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            order.compare(&Coord(1, 2, 3), &Coord(3, 1, 3)),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            order.compare(&Coord(3, 1, 3), &Coord(3, 1, 3)),
            std::cmp::Ordering::Equal
        );

        let order = AxisOrder::new(2, 1, 0);
        assert_eq!(
            order.compare(&Coord(1, 2, 3), &Coord(3, 1, 3)),
            std::cmp::Ordering::Greater
        );
    }
}
