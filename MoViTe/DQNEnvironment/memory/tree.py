Đã dùng 74% bộ nhớ … 
Nếu hết dung lượng lưu trữ, bạn sẽ không thể tạo, chỉnh sửa và tải tệp lên. Sử dụng 100 GB dung lượng với giá 45.000 ₫ 11.250 ₫ trong 1 tháng.

# The ‘sum-tree’ data structure used here is very similar in spirit to the array representation
# of a binary heap. However, instead of the usual heap property, the value of a parent node is
# the sum of its children. Leaf nodes store the transition priorities and the internal nodes are
# intermediate sums, with the parent node containing the sum over all priorities, p_total. This
# provides a efficient way of calculating the cumulative sum of priorities, allowing O(log N) updates
# and sampling. (Appendix B.2.1, Proportional prioritization)

# Additional useful links
# Good tutorial about SumTree data structure:  https://adventuresinmachinelearning.com/sumtree-introduction-python/
# How to represent full binary tree as array: https://stackoverflow.com/questions/8256222/binary-tree-represented-using-array

import numpy as np

class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get_prio(self, data_idx):
        idx = data_idx + self.size - 1  # child index in tree array
        return self.nodes[idx]
    
    def get_node_idx(self, idx):
        return idx + self.size - 1
    
    def sample(self, batch_size):
        # idx_data = self.nodes[self.get_node_idx(0):self.get_node_idx(self.size)]
        prob_ = []
        
        start_node = self.get_node_idx(0)
        end_node = self.get_node_idx(self.size)

        total_ = 0
        
        for i in range(start_node, end_node):
            # print("Node i: ", self.nodes[i] / self.total)
            prob_.append(self.nodes[i] / self.total)
            total_ += self.nodes[i] / self.total

        print("missing part: ", 1 - total_)

        missing_part = 1 - total_
        prob_[-1] += missing_part

        choice = np.random.choice(np.arange(0, self.size), size=batch_size, replace=False, p=prob_)
        
        return choice
    
    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"