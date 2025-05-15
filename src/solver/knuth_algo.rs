use std::ops::{Index, IndexMut};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NodeId(usize);

#[derive(Clone, Debug, PartialEq, Eq)]
struct Node {
    up: NodeId,
    down: NodeId,
    left: NodeId,
    right: NodeId,

    row: usize,
    column: usize,
}

const ROOT_ROW: usize = !0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnuthAlgorithm {
    arena: Vec<Node>,

    // Number of nodes in each column (except the head row).
    num_nodes_in_column: Vec<usize>,

    heads: Vec<NodeId>,

    // There are two types of columns: board blocks and pieces.
    // - board blocks: 0 .. num_board_blocks
    // - pieces: num_board_blocks .. num_board_blocks + piece_counts.len()
    num_board_blocks: usize,
    remaining_blocks: usize,
    piece_counts: Vec<u32>,
}

impl Index<NodeId> for KnuthAlgorithm {
    type Output = Node;

    fn index(&self, index: NodeId) -> &Node {
        &self.arena[index.0]
    }
}

impl IndexMut<NodeId> for KnuthAlgorithm {
    fn index_mut(&mut self, index: NodeId) -> &mut Node {
        &mut self.arena[index.0]
    }
}

impl KnuthAlgorithm {
    pub fn new(
        num_board_blocks: usize,
        placements: &[(usize, Vec<usize>)],
        piece_counts: &[u32],
    ) -> KnuthAlgorithm {
        let num_columns = num_board_blocks + piece_counts.len();
        let mut ret = KnuthAlgorithm {
            arena: vec![],
            num_nodes_in_column: vec![0; num_columns],
            heads: vec![],
            num_board_blocks,
            remaining_blocks: num_board_blocks,
            piece_counts: piece_counts.to_vec(),
        };

        // Create the root nodes
        for i in 0..num_columns {
            let node = ret.new_head_node(i);
            ret.heads.push(node);
        }

        // Create other nodes
        for i in 0..placements.len() {
            let (piece_id, block_ids) = &placements[i];

            let piece_node = ret.new_node(i, num_board_blocks + *piece_id);
            let mut last_node = piece_node;
            for &b in block_ids {
                let block_node = ret.new_node(i, b);

                ret[last_node].right = block_node;
                ret[block_node].left = last_node;
                ret[piece_node].left = block_node;
                ret[block_node].right = piece_node;

                last_node = block_node;
            }
        }

        ret
    }

    fn new_head_node(&mut self, column: usize) -> NodeId {
        let id = self.arena.len();
        self.arena.push(Node {
            up: NodeId(id),
            down: NodeId(id),
            left: NodeId(id),
            right: NodeId(id),

            row: ROOT_ROW,
            column,
        });
        NodeId(id)
    }

    fn new_node(&mut self, row: usize, column: usize) -> NodeId {
        let id = self.arena.len();

        let head = self.heads[column];
        let cur_last = self[head].up;
        self.arena.push(Node {
            up: cur_last,
            down: head,
            left: NodeId(id),
            right: NodeId(id),

            row,
            column,
        });
        self[head].up = NodeId(id);
        self[cur_last].down = NodeId(id);

        self.num_nodes_in_column[column] += 1;
        NodeId(id)
    }

    fn select_row(&mut self, node: NodeId) {
        let mut cur = node;
        loop {
            let column = self[cur].column;
            if column >= self.num_board_blocks {
                // Piece node
                let piece_id = column - self.num_board_blocks;
                assert!(self.piece_counts[piece_id] > 0);

                self.piece_counts[piece_id] -= 1;
                if self.piece_counts[piece_id] == 0 {
                    self.purge_column(cur);
                }
            } else {
                // Board block node
                self.purge_column(cur);
                self.remaining_blocks -= 1;
            }

            // detach `cur`
            let up = self[cur].up;
            let down = self[cur].down;
            self[up].down = down;
            self[down].up = up;

            cur = self[cur].right;
            if cur == node {
                break;
            }
        }
    }

    fn unselect_row(&mut self, node: NodeId) {
        let mut cur = self[node].left;
        loop {
            // attach `cur`
            let up = self[cur].up;
            let down = self[cur].down;
            self[up].down = cur;
            self[down].up = cur;

            let column = self[cur].column;
            if column >= self.num_board_blocks {
                // Piece node
                let piece_id = column - self.num_board_blocks;
                if self.piece_counts[piece_id] == 0 {
                    self.unpurge_column(cur);
                }
                self.piece_counts[piece_id] += 1;
            } else {
                // Board block node
                self.unpurge_column(cur);
                self.remaining_blocks += 1;
            }

            if cur == node {
                break;
            }
            cur = self[cur].left;
        }
    }

    // Make rows containing a node from the column in which `node` belongs unavailable.
    fn purge_column(&mut self, node: NodeId) {
        let mut cur = self[node].down;
        let target_col = self[node].column;

        self.num_nodes_in_column[target_col] -= 1;
        while cur != node {
            if self[cur].row == ROOT_ROW {
                cur = self[cur].down;
                continue;
            }

            let mut c = self[cur].right;
            while c != cur {
                let up = self[c].up;
                let down = self[c].down;
                self[up].down = down;
                self[down].up = up;

                let col = self[c].column;
                self.num_nodes_in_column[col] -= 1;

                c = self[c].right;
            }

            self.num_nodes_in_column[target_col] -= 1;
            cur = self[cur].down;
        }
    }

    // Revert the changes by `purge_column`.
    fn unpurge_column(&mut self, node: NodeId) {
        let mut cur = self[node].up;
        let target_col = self[node].column;

        self.num_nodes_in_column[target_col] += 1;
        while cur != node {
            if self[cur].row == ROOT_ROW {
                cur = self[cur].up;
                continue;
            }

            let mut c = self[cur].left;
            while c != cur {
                let up = self[c].up;
                let down = self[c].down;
                self[up].down = c;
                self[down].up = c;

                let col = self[c].column;
                self.num_nodes_in_column[col] += 1;

                c = self[c].left;
            }

            self.num_nodes_in_column[target_col] += 1;
            cur = self[cur].up;
        }
    }

    fn search_internal(&mut self, selected_rows: &mut Vec<usize>, answers: &mut Vec<Vec<usize>>) {
        if self.remaining_blocks == 0 {
            // Found a solution
            answers.push(selected_rows.clone());
            return;
        }

        let mut pivot = None;
        let mut pivot_cnt = self.arena.len(); // ???
        for i in 0..self.num_board_blocks {
            if self.num_nodes_in_column[i] > 0 {
                if self.num_nodes_in_column[i] < pivot_cnt {
                    pivot_cnt = self.num_nodes_in_column[i];
                    pivot = Some(i);
                }
            }
        }

        if pivot.is_none() {
            return;
        }

        let pivot = pivot.unwrap();
        let head = self.heads[pivot];

        let mut cur = self[head].down;
        while cur != head {
            selected_rows.push(self[cur].row);
            self.select_row(cur);

            self.search_internal(selected_rows, answers);

            self.unselect_row(cur);
            selected_rows.pop();

            cur = self[cur].down;
        }
    }

    pub fn search(&mut self) -> Vec<Vec<usize>> {
        let mut selected_rows = vec![];
        let mut answers = vec![];
        self.search_internal(&mut selected_rows, &mut answers);
        answers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn non_head_nodes(knuth_algo: &KnuthAlgorithm) -> Vec<NodeId> {
        let mut ret = vec![];
        for i in 0..knuth_algo.arena.len() {
            if knuth_algo.arena[i].row != ROOT_ROW {
                ret.push(NodeId(i));
            }
        }
        ret
    }

    #[test]
    fn test_knuth_algo_operators() {
        let knuth_algo = KnuthAlgorithm::new(
            4,
            &[
                (0, vec![0, 1, 2]),
                (0, vec![0, 3]),
                (1, vec![1, 2, 3]),
                (1, vec![0, 2]),
                (1, vec![1, 2]),
            ],
            &[1, 1],
        );

        let non_heads = non_head_nodes(&knuth_algo);
        assert_eq!(non_heads.len(), 17);

        // `purge_column` should be reverted by `unpurge_column`.
        for &pivot in &non_heads {
            let mut tmp = knuth_algo.clone();
            tmp.purge_column(pivot);
            tmp.unpurge_column(pivot);
            assert_eq!(knuth_algo, tmp);
        }

        // `select_row` should be reverted by `unselect_row`.
        for &pivot in &non_heads {
            let mut tmp = knuth_algo.clone();
            tmp.select_row(pivot);
            tmp.unselect_row(pivot);
            assert_eq!(knuth_algo, tmp);
        }
    }

    fn normalize_answers(answers: &mut Vec<Vec<usize>>) {
        for a in answers.iter_mut() {
            a.sort();
        }
        answers.sort();
    }

    #[test]
    fn test_knuth_algo_search() {
        let mut knuth_algo = KnuthAlgorithm::new(
            4,
            &[
                (0, vec![0, 1, 2]),
                (0, vec![0, 3]),
                (1, vec![1, 2, 3]),
                (1, vec![0, 2]),
                (1, vec![1, 2]),
            ],
            &[1, 1],
        );

        let initial = knuth_algo.clone();
        let mut answers = knuth_algo.search();
        assert_eq!(initial, knuth_algo);

        normalize_answers(&mut answers);

        assert_eq!(answers, vec![vec![1, 4]]);
    }

    #[test]
    fn test_knuth_algo_piece_limit() {
        {
            let mut knuth_algo = KnuthAlgorithm::new(
                4,
                &[
                    (0, vec![0, 1, 2]),
                    (0, vec![0, 3]),
                    (0, vec![1, 2, 3]),
                    (0, vec![0, 2]),
                    (0, vec![1, 2]),
                ],
                &[1],
            );

            let initial = knuth_algo.clone();
            let answers = knuth_algo.search();
            assert_eq!(initial, knuth_algo);
            assert_eq!(answers.len(), 0);
        }
        {
            let mut knuth_algo = KnuthAlgorithm::new(
                4,
                &[
                    (0, vec![0, 1, 2]),
                    (0, vec![0, 3]),
                    (0, vec![1, 2, 3]),
                    (0, vec![0, 2]),
                    (0, vec![1, 2]),
                ],
                &[2],
            );

            let initial = knuth_algo.clone();
            let mut answers = knuth_algo.search();
            assert_eq!(initial, knuth_algo);
            normalize_answers(&mut answers);

            assert_eq!(answers, vec![vec![1, 4]]);
        }
    }
}
