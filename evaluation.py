import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class Evaluation(nn.Module):
    def __init__(self, df_test, df_query, dataloader_test, dataloader_query, cuda):
        self.test_labels = np.array(df_test['label'])
        self.test_cameras = np.array(df_test['camera'])
        self.query_labels = np.expand_dims(np.array(df_query['label']), 1)
        self.query_cameras = np.expand_dims(np.array(df_query['camera']), 1)
        self.distractors = self.test_labels == 0
        self.junk = self.test_labels == -1
        
        self.test_labels = torch.Tensor(self.test_labels)
        self.test_cameras = torch.Tensor(self.test_cameras)
        self.query_labels = torch.Tensor(self.query_labels)
        self.query_cameras = torch.Tensor(self.query_cameras)
        self.distractors = torch.Tensor(self.distractors.astype(int))
        self.junk = torch.Tensor(self.junk.astype(int))
        
        if cuda:
            self.test_labels = self.test_labels.cuda()
            self.test_cameras = self.test_cameras.cuda()
            self.query_labels = self.query_labels.cuda()
            self.query_cameras = self.query_cameras.cuda()
            self.distractors = self.distractors.cuda()
            self.junk = self.junk.cuda()

        self.dataloader_test = dataloader_test
        self.dataloader_query = dataloader_query
        self.cuda = cuda
        
    def ranks_map(self, model, maxrank, remove_fc=False, features_normalized=True):
        if remove_fc:
            model = nn.Sequential(*list(model.children())[:-1])
        test_descriptors = self.descriptors(self.dataloader_test, model)
        query_descriptors = self.descriptors(self.dataloader_query, model)
        
        # cosine distances between query and test descriptors
        if features_normalized:
            dists = 1 - torch.mm(query_descriptors, test_descriptors.transpose(1, 0))
        else:
            dists = torch.mm(query_descriptors, test_descriptors.transpose(1, 0))
            dists = dists / torch.norm(query_descriptors, 2, 1).unsqueeze(1)
            dists = dists / torch.norm(test_descriptors, 2, 1)
            dists = 1 - dists
            
        dists_sorted, dists_sorted_inds = torch.sort(dists)
            
        # sort test data by indices which sort distances
        def sort_by_dists_inds(data):
            return torch.gather(data.repeat(self.query_labels.shape[0], 1), 1, dists_sorted_inds)
        
        test_sorted_labels = sort_by_dists_inds(self.test_labels)
        test_sorted_cameras = sort_by_dists_inds(self.test_cameras)
        sorted_distractors = sort_by_dists_inds(self.distractors).byte()
        sorted_junk = sort_by_dists_inds(self.junk).byte()
        
        # junk are not taken into account unlike distractors, so junk cumulative sum is calculated to be used later
        sorted_junk = (sorted_junk | 
                       (test_sorted_labels == self.query_labels) & 
                       (test_sorted_cameras == self.query_cameras))
        junk_cumsum = torch.cumsum(sorted_junk.int(), 1)
        
        # indices where query labels equal test labels without distractors and junk
        eq_inds = torch.nonzero(~sorted_distractors & ~sorted_junk & (self.query_labels == test_sorted_labels))
        eq_inds_rows = eq_inds[:, 0].long()
        eq_inds_cols = eq_inds[:, 1].long()
        eq_inds_first = np.unique(eq_inds_rows.cpu().numpy(), return_index=True)[1]
        # subtract junk cumsum from columns of indices
        eq_inds_cols_nojunk = (eq_inds_cols - junk_cumsum[eq_inds_rows, eq_inds_cols]).cpu().numpy()

        ranks = self.ranks(maxrank, eq_inds_first, eq_inds_cols_nojunk)
        mAP = self.mAP(eq_inds_first, eq_inds_cols_nojunk)

        return ranks, mAP
    
    def descriptors(self, dataloder, model):
        result = torch.FloatTensor()
        if self.cuda:
            result = result.cuda()
        for data in dataloder:
            if self.cuda:
                data = data.cuda()
            inputs = Variable(data)
            outputs = model(inputs)
            result = torch.cat((result, outputs.data), 0)

        return result

    def ranks(self, maxrank, eq_inds_first, eq_inds_cols):
        eq_inds_cols_first = eq_inds_cols[eq_inds_first]
        eq_inds_cols_first_maxrank = eq_inds_cols_first[eq_inds_cols_first < maxrank]
        ranks = np.zeros(maxrank)
        np.add.at(ranks, eq_inds_cols_first_maxrank, 1)
        ranks = np.cumsum(ranks)

        return ranks / self.query_labels.shape[0]
    
    def mAP(self, eq_inds_first, eq_inds_cols):
        labels_count = np.append(eq_inds_first[1:], eq_inds_cols.shape[0]) - eq_inds_first
        inds_start_repeat = np.repeat(eq_inds_first, labels_count)
        labels_count_repeat = np.repeat(labels_count, labels_count)
        average_precision = np.sum((np.arange(eq_inds_cols.shape[0]) - inds_start_repeat + 1) /
                                   (eq_inds_cols + 1) / 
                                   labels_count_repeat)

        return average_precision / self.query_labels.shape[0]
