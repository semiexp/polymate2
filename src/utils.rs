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

#[cfg(test)]
pub mod tests {
    use crate::shape::{Coord, Shape};

    pub fn shape_from_string(s: &str) -> Shape {
        let lines = s
            .trim()
            .lines()
            .map(|line| line.trim_start().split(" ").collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let num_layers = lines[0].len();
        let height = lines.len();
        let width = lines[0][0].len();

        let mut ret = Shape::new(
            vec![false; width * height * num_layers],
            Coord(num_layers as i32, height as i32, width as i32),
        );
        for y in 0..height {
            for z in 0..num_layers {
                assert_eq!(lines[y][z].len(), width);
                for (x, s) in lines[y][z].chars().enumerate() {
                    ret[(z, y, x)] = s == '#';
                }
            }
        }

        ret
    }

    pub fn shapes_from_strings(strings: &[&str]) -> Vec<Shape> {
        strings.iter().map(|s| shape_from_string(s)).collect()
    }

    #[test]
    fn test_shape_from_string() {
        // 2D shape
        {
            let expected =
                Shape::from_array_2d(vec![vec![true, true, true], vec![true, true, false]]);
            let actual = shape_from_string(
                "###
                 ##.",
            );

            assert_eq!(actual, expected);
        }

        // 3D shape
        {
            let expected = Shape::from_array_3d(vec![
                vec![vec![false, true], vec![false, false]],
                vec![vec![true, true], vec![true, false]],
            ]);
            let actual = shape_from_string(
                ".# ##
                 .. #.",
            );

            assert_eq!(actual, expected);
        }
    }
}
