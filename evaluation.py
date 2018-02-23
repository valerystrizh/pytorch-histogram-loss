import numpy as np
import os
import torch
import torch.nn as nn

from scipy.spatial.distance import cdist
from torch.autograd import Variable

class Evaluation(nn.Module):
    def __init__(self, df_test, df_query, dataloader_test, dataloader_query, cuda):
        self.test_labels = np.array(df_test['label'])
        self.test_cameras = np.array(df_test['camera'])
        self.query_labels = np.expand_dims(np.array(df_query['label']), 1)
        self.query_cameras = np.expand_dims(np.array(df_query['camera']), 1)
        self.distractors = np.array(df_test['label'] == 0)
        self.junk = np.array(df_test['label'] == -1)
        self.dataloader_test = dataloader_test
        self.dataloader_query = dataloader_query
        self.cuda = cuda
        
    def ranks_map(self, model, maxrank, remove_fc=False):
        if remove_fc:
            model = nn.Sequential(*list(model.children())[:-1])
        test_descr = self.descriptors(self.dataloader_test, model)
        query_descr = self.descriptors(self.dataloader_query, model)
        dists = cdist(query_descr, test_descr, 'cosine')
        dists_argsort = np.argsort(dists)
        test_sorted_labels = self.test_labels[dists_argsort]
        test_sorted_cameras = self.test_cameras[dists_argsort]
        sorted_distractors = self.distractors[dists_argsort]
        
        junk = (self.junk[dists_argsort] | 
               (test_sorted_labels == self.query_labels) & 
               (test_sorted_cameras == self.query_cameras))
        junk_cumsum = np.cumsum(junk, 1)
        
        eq_inds = np.where(~sorted_distractors & ~junk & (self.query_labels == test_sorted_labels))
        eq_inds_rows = eq_inds[0]
        eq_inds_cols = eq_inds[1]
        eq_inds_cols_nojunk = eq_inds_cols - junk_cumsum[eq_inds_rows, eq_inds_cols]

        ranks = self.ranks(maxrank, eq_inds_rows, eq_inds_cols_nojunk)
        mAP = self.mAP(eq_inds_rows, eq_inds_cols_nojunk)

        return ranks, mAP
    
    def descriptors(self, dataloder, model):
        result = []

        for data in dataloder:
            if self.cuda:
                inputs = Variable(data.cuda())
            else:
                inputs = Variable(data)

            outputs = model(inputs)
            result.extend(outputs.data.squeeze().cpu().numpy())

        return np.array(result)

    def ranks(self, maxrank, eq_inds_rows, eq_inds_cols_nojunk):
        eq_inds_first = np.unique(eq_inds_rows, return_index=True)[1]
        eq_inds_cols_first_nojunk = eq_inds_cols_nojunk[eq_inds_first]
        eq_inds_cols_first_nojunk_maxrank = eq_inds_cols_first_nojunk[eq_inds_cols_first_nojunk < maxrank]
        ranks = np.zeros(maxrank)
        np.add.at(ranks, eq_inds_cols_first_nojunk_maxrank, 1)
        ranks = np.cumsum(ranks)

        return ranks / self.query_labels.shape[0]
    
    def mAP(self, eq_inds_rows, eq_inds_cols_nojunk):
        eq_inds_unique = np.unique(eq_inds_rows, return_index=True)[1]
        labels_count = np.append(eq_inds_unique[1:], eq_inds_rows.shape[0]) - eq_inds_unique
        inds_start_repeat = np.repeat(eq_inds_unique, labels_count)
        labels_count_repeat = np.repeat(labels_count, labels_count)
        average_precision = np.sum((np.arange(eq_inds_rows.shape[0]) - inds_start_repeat + 1) /
                                   (eq_inds_cols_nojunk + 1) / labels_count_repeat)

        return average_precision / self.query_labels.shape[0]
