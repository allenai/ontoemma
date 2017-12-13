import numpy as np
import tqdm
import emma.constants as constants
from random import shuffle


# modified hungarian algorithm for bipartite assignment
class ModifiedHungarian:
    def __init__(self, mat):
        self.best_n = constants.MODH_KEEP_TOP_N_CANDIDATES
        self.transposed = False

        self.mat = mat

        self.row_num = len(mat)
        self.col_num = len(mat[0])

        if self.col_num > self.row_num:
            self.transposed = True
            self.mat = np.transpose(self.mat)

    def _get_bestn_per_row(self, mat):
        keep_n = dict()
        for row_id in range(self.row_num):
            row_sort = sorted(zip(range(self.col_num), mat[row_id]), key=lambda x: x[1])
            if row_sort[0][0] != 1.0:
                keep_n[row_id] = [[i, True] for i, c in row_sort[:self.best_n] if c < 1.0]
        return keep_n

    def _get_first_unassigned(self, l):
        """
        Get first unassigned item in keep list
        :param l:
        :return:
        """
        for i, v in enumerate(l):
            if v[1]:
                return i
        return -1

    @staticmethod
    def _obliterate_ind(ind, ind_dict):
        for k, v in ind_dict.items():
            for i, p in enumerate(v):
                if p[0] == ind:
                    ind_dict[k][i][1] = False
        return ind_dict

    def _flatten(self, l):
        return [item for sublist in l for item in sublist]

    def _get_min_remaining_cost(self, ind_dict):
        d_pairs = self._flatten([[(k, i[0]) for i in v if i[1]] for k, v in ind_dict.items()])
        d_costs = [self.mat[r_ind][c_ind] for (r_ind, c_ind) in d_pairs]
        if d_costs:
            return min(d_costs)
        else:
            return 0.0

    def _normalize_mat(self):
        self.mat[self.mat < 0.0] = 0.0
        self.mat[self.mat > 1.0] = 1.0
        return

    def _update_costs(self, min_cost, ind_dict):
        """
        Subtract min_cost from all dict entries that are unassigned
        Add min_cost from all dict entries that have been assigned
        :param min_cost:
        :param ind_dict:
        :return:
        """
        for k, v in ind_dict.items():
            for i, p in enumerate(v):
                if p[1]:
                    self.mat[k][v[i][0]] -= min_cost
                else:
                    self.mat[k][v[i][0]] += min_cost
        self._normalize_mat()
        return

    def _reset_assignments(self, ind_dict):
        for k in ind_dict:
            for i in range(self.best_n):
                ind_dict[k][i][1] = True
        return ind_dict

    def _process_transpose(self, a_mat):
        if self.transposed:
            return [(c_ind, r_ind) for (r_ind, c_ind) in a_mat]
        else:
            return a_mat

    def _compute_total_cost(self, a_mat):
        total = 0.0
        for r_ind, c_ind in a_mat:
            total += self.mat[r_ind][c_ind]
        return total

    def compute(self):
        """
        Compute match indices over input cost matrix
        :param mat: cost matrix
        :return:
        """
        iter_unchanged = 0

        row_top_n = self._get_bestn_per_row(self.mat)

        row_assignments = []
        best_assignment = []
        lowest_cost = float(max([self.row_num, self.col_num]))

        for num_iter in tqdm.tqdm(range(100), total=100, desc="Iteration"):
            # process row side
            rand_row = list(range(self.row_num))
            shuffle(rand_row)
            for row_ind in rand_row:
                if row_ind in row_top_n:
                    col_opts = row_top_n[row_ind]
                    assign = self._get_first_unassigned(col_opts)
                    if assign >= 0:
                        col_ind = col_opts[assign][0]
                        row_assignments.append((row_ind, col_ind))
                        row_top_n = self._obliterate_ind(col_ind, row_top_n)

            temp_assignment = set(row_assignments)

            total_cost = self._compute_total_cost(temp_assignment)

            if total_cost < lowest_cost:
                best_assignment = temp_assignment
                lowest_cost = total_cost

            if total_cost >= lowest_cost:
                iter_unchanged += 1

            if iter_unchanged > 10:
                return self._process_transpose(best_assignment)

            min_cost = self._get_min_remaining_cost(row_top_n)
            self._update_costs(min_cost, row_top_n)
            row_top_n = self._get_bestn_per_row(self.mat)

        return self._process_transpose(best_assignment)