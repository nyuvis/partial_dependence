# -*- coding: utf-8 -*-
"""
partial_dependence is a library for plotting partial dependency patterns
of machine learning classfiers. Partial dependence measures the
prediction change when changing one or more input features. We will
focus only on 1D partial dependency plots. For each instance in the data
we can plot the prediction change as we change a single feature in a
defined sample range. Then we cluster similar plots, e.g., instances
reacting similarly value changes, to reduce clutter. The technique is a
black box approach to recognize sets of instances where the model makes
similar decisions.
"""
from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import time
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
import scipy.interpolate as si
import sys
from matplotlib.legend import Legend
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec






try:
    unicode = unicode
except NameError:
    # python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str, bytes)
else:
    # python 2
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring

    
__version__ = "0.0.4"

class PdpCurves(object):


    def __init__(self, preds):

        self._preds = preds
        self._ixs = np.arange(self._preds.shape[0])

        self._dm = None
        self.r_param = None


    def copy(self):

        return self.__copy__()

    def __copy__(self):

      res = type(self)(self._preds)
      res.__dict__.update(self.__dict__)

      return res

    def get_mean_distances(self, cluster_A, cluster_B):

        dist_mtrx = self._dm
        real_A = cluster_A[1]
        real_B = cluster_B[1]
        lab_A = cluster_A[0]
        lab_B = cluster_B[0]

        if lab_A == lab_B:
            return 1.1

        inst_A_list = real_A.get_ixs()
        inst_B_list = real_B.get_ixs()

        distance_list = []

        for a in inst_A_list:
            for b in inst_B_list:
                distance_list.append(dist_mtrx[a,b])

        return np.mean(distance_list)

    def split(self,labels_cluster):
        if labels_cluster is None:
            return [ (None, self.copy()) ]

        def get_slice(c, lbl):
            c._preds = self._preds[labels_cluster == lbl, :]
            if self._dm is not None:
                c._dm = self._dm[labels_cluster == lbl, :][:, labels_cluster == lbl]
            c._ixs = self._ixs[labels_cluster == lbl]
            return c

        def get_macro_dist(clusters_input):
            
            pairs_of_clusters = []
            for comb in combinations(list_of_clusters, 2):
                pairs_of_clusters.append(comb)

            num_cluster_here = len(list_of_clusters)
            macro_dist_matrix = np.zeros((num_cluster_here,num_cluster_here))

            for i in list_of_clusters:
                for j in list_of_clusters:
                    macro_dist_matrix[i,j] = self.get_mean_distances(clusters_input[i],
                                                                     clusters_input[j])

            return macro_dist_matrix


        list_of_clusters = np.unique(labels_cluster)

        curves_list = []
        for lbl in list_of_clusters:
            the_slice = get_slice(self.copy(), lbl)
            curves_list.append((lbl, the_slice)) 

        return (curves_list,get_macro_dist(curves_list))

    def get_ixs(self):
        return self._ixs


    def get_preds(self):
        return self._preds

    def get_dm(self, never_splom = False):
        if self._dm is None:
            self._dm = self._compute_dm()
        return self._dm

    def _compute_dm(self):
        preds = self._preds
        lenTest = len(preds)

        def rmse(curve1, curve2):
            return np.sqrt(((curve1 - curve2) ** 2).mean())

        def lb_keogh(s1, s2, r):
            LB_sum=0
            for (ind, i) in enumerate(s1):
                lower_bound=min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
                upper_bound=max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
                if i > upper_bound:
                    LB_sum = LB_sum + (i - upper_bound) ** 2
                elif i < lower_bound:
                    LB_sum = LB_sum + (i - lower_bound) ** 2
            return np.sqrt(LB_sum)
        
        if self.r_param is None:
            lb_keogh_bool = False
        else:
            lb_keogh_bool = True

        list_of_test_indexes = list(range(lenTest))
        pairs_of_curves = []
        for comb in combinations(list_of_test_indexes, 2):
            pairs_of_curves.append(comb)

        distance_matrix = np.zeros((lenTest, lenTest))

        for pair in pairs_of_curves:
            i = pair[0]
            j = pair[1]

            if lb_keogh_bool:
                distance = lb_keogh(preds[i], preds[j], self.r_param)
            else:
                distance = rmse(preds[i], preds[j])

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
            distance_matrix[i, i] = 0.0

        return distance_matrix

    def get_keogh_radius(self):
        return self.r_param

    def set_keogh_radius(self, r_param):
        self._dm = None
        self.r_param = r_param







class Pdp2DCurves(object):


    def __init__(self, raw, original_ps, ixs, feat_y, feat_x):

        self._raw = raw
        self._origs = original_ps
        self._ixs = ixs
        self.A = feat_y
        self.B = feat_x

        self._dm = None

        self.sample_A = None
        self.sample_B = None
        self.heat_map = None



    def get_sample_A(self):
        return self.sample_A

    def get_sample_B(self):
        return self.sample_B

    def get_A(self):
        return self.A

    def get_B(self):
        return self.B

    def compute_heatmap(self):
        heatmap_datas = []
        num_samples = self._raw.shape[1]
        num_inst = len(self._ixs)
        for i in range(num_inst):
            single_pdp = self._raw[i]
            orig_pred = self._origs[i]
            heatmap_datas.append(single_pdp-orig_pred)
        
        heatmap_data_final = []
        for i in range(num_samples):
            heatmap_data_final.append(list(np.zeros(num_samples)))
            
        for heat in heatmap_datas:
            heatmap_data_final+=heat
                    
        heatmap_data_final = heatmap_data_final / num_inst
            
        return heatmap_data_final

    def get_data(self):
        if self.heat_map is None:
            self.heat_map = self.compute_heatmap()
        return self.heat_map

    def get_preds(self):
        return self._raw

    def get_origs(self):
        return self._origs

    def get_ixs(self):
        return self._ixs

    def set_sample_data(self,A_sample,B_sample):
        self.sample_A = A_sample
        self.sample_B = B_sample

    def get_sample_data(self):

        return self.sample_A, self.sample_B

    def copy(self):

        return self.__copy__()

    def __copy__(self):

        res = type(self)(self._raw, self._origs, self._ixs, self.A, self.B)
        res.__dict__.update(self.__dict__)

        return res

    def swap_features(self):

        swapped_copy = self.copy()

        swapped_copy.heat_map = self.get_data().transpose()

        raws = []
        for r in self._raw:
           raws.append(r.transpose()) 
        raws = np.array(raws)

        swapped_copy._raw = raws

        swapped_copy.A = self.B
        swapped_copy.B = self.A

        swapped_copy.sample_A = self.sample_B
        swapped_copy.sample_B = self.sample_A
        return swapped_copy

    def get_dm(self, is_splom = False):
        if self._dm is None:
            self._dm = self._compute_dm(is_splom)
        return self._dm

    def set_dm(self,matrix):
        self._dm = matrix

    def _compute_dm(self, splom_flag = False):
        preds = self._raw
        lenTest = len(self._ixs)

        def rmse(curve1, curve2, do_you_splom = False):
            if do_you_splom:
                return ((curve1 - curve2) ** 2).sum()
            return np.sqrt(((curve1 - curve2) ** 2).mean())


        list_of_test_indexes = list(range(lenTest))

        pairs_of_curves = []
        for comb in combinations(list_of_test_indexes, 2):
            pairs_of_curves.append(comb)

        distance_matrix = np.zeros((lenTest, lenTest))

        for pair in pairs_of_curves:
            i = pair[0]
            j = pair[1]

            distance = rmse(preds[i].reshape((-1,)), preds[j].reshape((-1,)),splom_flag)

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
            distance_matrix[i, i] = 0.0

        return distance_matrix

    def get_mean_distances(self, cluster_A, cluster_B):

        dist_mtrx = self._dm
        real_A = cluster_A[1]
        real_B = cluster_B[1]
        lab_A = cluster_A[0]
        lab_B = cluster_B[0]

        if lab_A == lab_B:
            return 1.1

        inst_A_list = real_A._index_ixs
        inst_B_list = real_B._index_ixs

        distance_list = []

        for a in inst_A_list:
            for b in inst_B_list:
                distance_list.append(dist_mtrx[a,b])

        return np.mean(distance_list)

    def split(self,labels_cluster):

        if labels_cluster is None:
            return [ (None, self.copy()) ]

        def get_slice(c, lbl):
            c._raw = self._raw[labels_cluster == lbl, :]
            if self._dm is not None:
                c._dm = self._dm[labels_cluster == lbl, :][:, labels_cluster == lbl]
            c._ixs = self._ixs[labels_cluster == lbl]
            c._origs = self._origs[labels_cluster == lbl]
            c._index_ixs = np.array(range(len(self._ixs)))[labels_cluster == lbl]

            c.heat_map = c.compute_heatmap()

            return c

        def get_macro_dist(clusters_input):
            
            pairs_of_clusters = []
            for comb in combinations(list_of_clusters, 2):
                pairs_of_clusters.append(comb)

            num_cluster_here = len(list_of_clusters)
            macro_dist_matrix = np.zeros((num_cluster_here,num_cluster_here))

            for i in list_of_clusters:
                for j in list_of_clusters:
                    macro_dist_matrix[i,j] = self.get_mean_distances(clusters_input[i],
                                                                     clusters_input[j])

            return macro_dist_matrix


        list_of_clusters = np.unique(labels_cluster)

        curves_list = []
        for lbl in list_of_clusters:
            the_slice = get_slice(self.copy(), lbl)
            curves_list.append((lbl, the_slice)) 

        return (curves_list,get_macro_dist(curves_list))

class PartialDependence(object):

    '''

    Initialization Parameters
    -------------------------

    df_test: pandas.DataFrame
        (REQUIRED) dataframe containing only the features values for each instance in the test-set.

    model: Python object
        (REQUIRED) Trained classifier as an object with the following properties:
        The object must have a method predict_proba(X) 
        which takes a numpy.array of shape (n, num_feat) 
        as input and returns a numpy.array of shape (n, len(class_array)).

    class_array: list of strings
        (REQUIRED) all the classes name in the same order as the predictions returned by predict_proba(X).

    class_focus: string
        (REQUIRED) class name of the desired partial dependence

    num_samples: integer value
        (OPTIONAL)[default = 100] number of desired samples. Sampling a feature is done with:
            numpy.linspace(min_value,max_value,num_samples)
        where the bounds are related to min and max value for that feature in the test-set.

    scale: float value
        (OPTIONAL)[default = None] scale parameter vector for normalization.

    shift: float value
        (OPTIONAL)[default = None] shift parameter vector for normalization.
        
            If you need to provide your data to the model in normalized form, 
            you have to define scale and shift such that:
                transformed_data = (original_data + shift)*scale
            where shift and scale are both numpy.array of shape (1,num_feat).
            If the model uses directly the raw data in df_test without any transformation, 
            do not insert any scale and shift parameters.


    '''


    def __init__(self, 
                  df_test,
                  model,
                  class_array,
                  class_focus,
                  num_samples=100,
                  scale=None,
                  shift=None):

        self.df = df_test
        self.mdl = model
        self.cls_arr = class_array
        self.cls_fcs = class_focus
        self.n_smpl = int(np.floor(num_samples/2)*2)
        self.scl = scale
        self.shft = shift
        
        self._compute_sampling()
        
    def _compute_sampling(self):

        df_test = self.df
        model = self.mdl
        class_array = self.cls_arr
        class_focus = self.cls_fcs
        num_samples = self.n_smpl
        scale = self.scl
        shift = self.shft

        def check_denormalization(scale_check, shift_check):
            if scale_check is None or shift_check is None:
                de_norm = False
            else:
                de_norm = True
            return de_norm

        de_norm_bool = check_denormalization(scale, shift)

        data_set_pred_index = class_array.index(class_focus)
        lenTest = len(df_test)
        num_feat = len(df_test.columns)
        
        #ylabel = [df_test.columns[-1]]

        xlabel = list(df_test.columns)
        x_array = df_test[xlabel].as_matrix()

        #y_array = df_test[ylabel].as_matrix()
        #labs = [True if l[0] in [class_array[0]] else False for l in y_array]

        if de_norm_bool:
            x_array = (x_array + shift) * scale

        pred = model.predict_proba(x_array)
        original_preds = np.array([ x[data_set_pred_index] for x in pred ])
        #thresh = get_roc_curve(original_preds,labs)["score"]

        dictLabtoIndex = {}
        for (ix, laballa) in enumerate(xlabel):
            dictLabtoIndex[laballa] = ix

        df_features = pd.DataFrame(columns=[ "max","min","mean","sd" ], index=xlabel)

        means = []
        stds = []
        mins = []
        maxs = []
        for laballa in xlabel:

            THErightIndex = dictLabtoIndex[laballa]

            if de_norm_bool:
                vectzi = (list(df_test[laballa]) + shift[0][THErightIndex]) * scale[0][THErightIndex]
            else:
                vectzi = list(df_test[laballa])

            mean = np.mean(vectzi)
            maxim = max(vectzi)
            minion = min(vectzi)
            standin = np.std(vectzi)
            means.append(mean)
            stds.append(standin)
            mins.append(minion)
            maxs.append(maxim)

        df_features["max"] = maxs
        df_features["min"] = mins
        df_features["mean"] = means 
        df_features["sd"] = stds
        num_feat = len(xlabel)

        df_sample = pd.DataFrame(columns=xlabel)

        eps = 0
        for laballa in xlabel:
            lower_bound = df_features["min"][laballa] - eps
            higher_bound = df_features["max"][laballa] + eps
            #bound = df_features["mean"][laballa] + 2*df_features["sd"][laballa]
            df_sample[laballa] = np.linspace(lower_bound, higher_bound, num_samples)

        changing_rows = np.copy(x_array)

        # local sampling


        def local_sampling (fix, chosen_row):
            base_value = chosen_row[dictLabtoIndex[fix]]
            samplesLeft = list(np.linspace(base_value-1, base_value, int(num_samples / 2) + 1))
            samplesRight = list(np.linspace(base_value, base_value + 1, int(num_samples / 2) + 1))
            samples = samplesLeft + samplesRight
            divisor = int(num_samples / 2 + 1)
            final_samples = samples[:divisor - 1] + [ base_value ] + samples[divisor + 1:]
            return final_samples

        list_of_df_local_sampling = []

        for i in range(lenTest):
            r = changing_rows[i]
            df = pd.DataFrame()

            for lab in xlabel:
                df[lab] = local_sampling (lab,r)

            list_of_df_local_sampling.append(df)
        
        self.changing_rows = changing_rows
        self.dictLabtoIndex = dictLabtoIndex
        self.original_preds = original_preds
        self.num_feat = num_feat
        self.lenTest = lenTest
        self.data_set_pred_index = data_set_pred_index
        self.df_sample = df_sample
        self.df_features = df_features
        self.de_norm_bool = de_norm_bool
        self.list_local_sampling = list_of_df_local_sampling
        
    def pdp(self, fix, chosen_row=None,batch_size=0):

        """
        By choosing a feature and changing it in the sample range, 
        for each row in the test-set we can create num_samples different versions of the original instance.
        Then we are able to compute prediction values for each of the different vectors.
        pdp() initialize and returns a python object from the class PdpCurves containing such predictions values.

        Parameters
        ----------

        fix : string
            (REQUIRED) The name of feature as reported in one of the df_test columns.
       
        chosen_row : numpy.array of shape (1,num_feat)
            (OPTIONAL) [default = None] A custom row, defined by the user, used to test or compare the results.
            For example you could insert a row with mean values in each feature.

        batch_size: integer value
            (OPTIONAL) [default = 0] The batch size is required when the size ( num_rows X num_samples X num_feat ) becomes too large.
            In this case you might want to compute the predictions in chunks of size batch_size, to not run out of memory.
            If batch_size = 0, then predictions are computed with a single call of model.predict_proba().
            Otherwise the number of calls is automatically computed and it will depend on the user-defined batch_size parameter.
            In order to not split up predictions relative to a same instance in different chunks, 
            batch_size must be greater or equale to num_samples.

        Returns
        -------

        curves_returned: python object
            (ALWAYS) An itialized object from the class PdpCurves.
            It contains all the predictions obtained from the different versions of the test instances, stored in matrix_changed_rows.

        chosen_row_preds: numpy.array of shape (1, num_samples)
            (IF REQUESTED) If chosen_row_alterations was supplied by the user, we also return this array, otherwise just pred_matrix is returned.
            It contains all the different predictions obtained from the different versions of custom chosen_row, stored in chosen_row_alterations.          


        """
            
        rows = self.changing_rows
        dictLabtoIndex = self.dictLabtoIndex
        num_feat = self.num_feat
        df_sample = self.df_sample
        num_samples = self.n_smpl
        
        self.the_feature = fix

        
        def pred_comp_all(self, matrix_changed_rows, chosen_row_alterations_input=None, batch_size_input=0):
            

            model = self.mdl
            #num_samples = self.n_smpl

            def compute_pred(self, matrix_changed_rows, chosen_row_alterations_sub_funct=None):
                #t = time.time()
                #num_feat = self.num_feat
                data_set_pred_index = self.data_set_pred_index

                num_rows= len(matrix_changed_rows)
                pred_matrix = np.zeros((num_rows, num_samples))
                matrix_changed_rows = matrix_changed_rows.reshape((num_rows * num_samples, num_feat))
                ps = model.predict_proba(matrix_changed_rows)
                ps = [ x[data_set_pred_index] for x in ps ]
                k = 0
                for i in range(0, num_rows * num_samples):
                    if i % num_samples == 0:
                        pred_matrix[k] = ps[i:i + num_samples]
                        k += 1

                curves_returned = PdpCurves(pred_matrix)

                if chosen_row_alterations_sub_funct is not None:
                    chosen_row_preds = model.predict_proba(chosen_row_alterations_sub_funct)
                    chosen_row_preds = np.array([ x[data_set_pred_index] for x in chosen_row_preds ])
                    return curves_returned, chosen_row_preds
                return curves_returned


            def compute_pred_in_chunks(self, matrix_changed_rows, number_all_preds_in_batch, chosen_row_alterations_sub_funct=None):

                #t = time.time()
                num_feat = self.num_feat
                data_set_pred_index = self.data_set_pred_index

                num_rows= len(matrix_changed_rows)
                pred_matrix = np.zeros((num_rows, num_samples))

                #number_all_preds_in_batch = 1000

                if number_all_preds_in_batch < num_samples:
                    print ("Error: batch size cannot be less than sample size.")
                    return np.nan

                if number_all_preds_in_batch > num_samples*num_rows:
                    print ("Error: batch size cannot be greater than total size (num. of instances X num. of samples).")
                    return np.nan




                num_of_instances_in_batch = int( np.floor( number_all_preds_in_batch / num_samples ) )
                how_many_calls = int( np.ceil( num_rows / num_of_instances_in_batch ) )
                residual = num_of_instances_in_batch * how_many_calls - num_rows
                num_of_instances_in_last_batch = num_of_instances_in_batch - residual

                #print ("num_of_instances_in_batch"  ,  num_of_instances_in_batch)
                #print ( "how_many_calls" , how_many_calls, type(how_many_calls) )
                #print ( "residual" , residual)
                #print ( "num_of_instances_in_last_batch" , num_of_instances_in_last_batch)


                for i in range(0, how_many_calls):
                    #if float(i+1)%1000==0:
                        #print ("---- loading preds: ", np.round(i/float(num_rows),decimals=4)*100,"%")
                        #print ("------ elapsed: ",int(int(time.time()-t)/60), "m")


                    low_bound_index = i*num_of_instances_in_batch
                    high_bound_index = low_bound_index + num_of_instances_in_batch


                    if i == how_many_calls -1 and residual != 0:
                        high_bound_index = low_bound_index + num_of_instances_in_last_batch

                    matrix_batch = matrix_changed_rows[low_bound_index:high_bound_index]

                    pred_matrix_batch = compute_pred(self,matrix_batch).get_preds()

                    pred_matrix[low_bound_index:high_bound_index] = np.copy(pred_matrix_batch)

                curves_returned = PdpCurves(pred_matrix)

                if chosen_row_alterations_sub_funct is not None:
                    chosen_row_preds = model.predict_proba(chosen_row_alterations_sub_funct)
                    chosen_row_preds = np.array([x[data_set_pred_index] for x in chosen_row_preds])
                    return curves_returned, chosen_row_preds
                return curves_returned

            if batch_size_input != 0:

                return compute_pred_in_chunks(self, matrix_changed_rows, 
                                                number_all_preds_in_batch = batch_size_input, 
                                                chosen_row_alterations_sub_funct = chosen_row_alterations_input)

            return compute_pred(self, matrix_changed_rows, 
                                chosen_row_alterations_sub_funct = chosen_row_alterations_input)


        num_rows = len(rows)
        new_matrix_f = np.zeros((num_rows, num_samples, num_feat))
        sample_vals = df_sample[fix]
        depth_index = 0
        for r in rows:
            index_height = 0
            for v in sample_vals:
                new_r = np.copy(r)
                new_r[dictLabtoIndex[fix]] = v
                new_matrix_f[depth_index][index_height] = new_r
                index_height += 1
            depth_index += 1

        if chosen_row is not None:

            chosen_row_alterations = []

            for v in sample_vals:
                arow = np.copy(chosen_row)
                arow[dictLabtoIndex[fix]] = v
                chosen_row_alterations.append(arow)
            
            chosen_row_alterations = np.array(chosen_row_alterations)

            return pred_comp_all(
                self, 
                new_matrix_f, 
                chosen_row_alterations_input=chosen_row_alterations,
                batch_size_input=batch_size)

        return pred_comp_all(
            self, 
            new_matrix_f,
            batch_size_input=batch_size)

    def get_optimal_keogh_radius(self):

            """
            computes the optimal value for the parameter needed to compute the LB Keogh distance.
            It is computed given the sample values, the standard deviation, max and min.

            """

            the_feature = self.the_feature
            num_samples = self.n_smpl
            df_sample = self.df_sample
            df_features = self.df_features

            distance_between_2_samples = \
                (df_sample[the_feature][num_samples-1] - df_sample[the_feature][0]) / num_samples
            sugg_r = int(np.ceil(df_features["sd"][the_feature] / 10.0 / distance_between_2_samples))
            #print("Suggested r parameter:",sugg_r)
            #rWarpedUsed = int(input("warp parameter window:"))
            rWarpedUsed = sugg_r
            if rWarpedUsed == 0:
                rWarpedUsed = 1

            return rWarpedUsed


    
    def compute_clusters(self, curves, clust_number=5):

        """
        Produces a clustering on the instances of the test set based on the similrity of the predictions values from preds.
        The clustering is done with the agglomerative technique using a distance matrix.
        The distance is measured either with root mean square error (RMSE) or with dynamic time warping distance (DTW),
        depending on the user choice.

        Parameters
        ----------
        curves : python object
            (REQUIRED) Returned by previous function pdp() or pdp_2D() or get_data_splom().

        clust_number : integer value
            (OPTIONAL) [default = 5] The number of desired clusters.

        Returns
        -------

        the_list_sorted : list of size clust_number with tuples: ( label cluster (float), python object )
            (ALWAYS) Each element is an object from the class PdpCurves() or Pdp2DCurves() representing a different cluster.
            This list is sorted using the clustering distance matrix taking the biggest cluster first, 
            then the closest cluster by average distance next and so on.

        """
        num_feat = self.num_feat
        num_samples = self.n_smpl

        from_splom = False
        if type(curves) is dict:
            from_splom = True


        if not from_splom:

            distance_matrix = curves.get_dm(from_splom)

        else:

            size_matrix = len(curves[(0,1)].get_ixs())
            distance_matrix = np.zeros((size_matrix,size_matrix))

            for ij in curves:
                curves_from_grid = curves[ij]
                distance_matrix += curves_from_grid.get_dm(from_splom)

            num_squares_summed = (num_samples+1)**2*(num_feat**2-num_feat)/2
            distance_matrix = np.sqrt(distance_matrix/num_squares_summed)


        clust = AgglomerativeClustering(affinity = 'precomputed', n_clusters = clust_number, linkage = 'average')

        clust.fit(distance_matrix)

        labels_array = clust.labels_

        def split_and_sort (curves_obj):

            goal = curves_obj.split(labels_array)
            the_list = goal[0]
            the_matrix_for_sorting = goal[1]

            size = 0
            for cl in the_list:
                size_new = len(cl[1].get_ixs())
                
                if size_new > size:
                    size = size_new
                    label_biggest = cl[0]

            cluster_labels_still = list(range(len(the_list)))
            corder = [ label_biggest ]
            cluster_labels_still.remove(label_biggest)

            while len(corder)<len(the_list):

                mins = {}
                for c in corder:
                    full_distances = list(the_matrix_for_sorting[c,:])
                    distances = list(the_matrix_for_sorting[c,cluster_labels_still])

                    the_min = np.min(distances)
                    the_index = full_distances.index(min(distances))
                    mins[the_index] = the_min

                new_cord = min(mins, key=mins.get)

                corder.append(new_cord)
                cluster_labels_still.remove(new_cord)


            if len(np.unique(corder)) != len(the_list):
                print("Fatal Error.")
                
            the_list_sorted = []
            for c in corder:
                the_list_sorted.append(the_list[c])

            return the_list_sorted

        if not from_splom:
            return split_and_sort (curves)

        else:
            heats_list = []
            labels_list = []
            for i in range(clust_number):
                heats_list.append({})
                labels_list.append(None)

            for ij in curves:
                curves[ij].set_dm(distance_matrix)
                list_heatmaps = split_and_sort(curves[ij])

                for i in range(clust_number):
                    label_cluster = list_heatmaps[i][0]
                    heat_obj = list_heatmaps[i][1]
  
                    if labels_list[i] is None:
                       labels_list[i] = label_cluster

                    else:
                        if labels_list[i] != label_cluster:
                            print ("mismatching clusters",labels_list[i],"vs",label_cluster)
                            return

                    heats_list[i][ij] = heat_obj

            return [ (labels_list[i], heats_list[i]) for i in range(clust_number) ]







    def _back_to_the_original(self, data_this, fix):

        # private function to be able to plot the data values not in normalized form

        dictLabtoIndex = self.dictLabtoIndex
        de_norm = self.de_norm_bool
        scale = self.scl
        shift = self.shft

        if de_norm:
            integ = dictLabtoIndex[fix]
            data_that = data_this / scale[0][integ] - shift[0][integ]
        else:
            data_that = data_this
        return data_that


    def plot(self,
             curves_input,
             color_plot = None,
             thresh = 0.5,
             local_curves = True,
             chosen_row_preds_to_plot = None,
             plot_full_curves = False,
             plot_object = None,
             cell_view = False,
             path = None):



        """
        The visualization will display broad curves with color linked to different clusters.
        The orginal instances are displayed with coordinates (orginal feature value, original prediction) either as
        local curves or dots depending on the local_curves argument value.

        Parameters
        ----------
        
        curves : python object
            (REQUIRED) A python object from the class PdpCurves(). ( Returned by previous function pdp() )
            Otherwise a list of such python objects in tuples. ( Returned by previous function compute_clusters() )

        color_plot : string or list of strings
            (OPTIONAL) [default = None] The color for each cluster of instances. 
            If there is no clustering or just a single cluster provide just a string with the desire color.

        thresh: float value
            (OPTIONAL) [default = 0.5]  The threshold is displayed as a red dashed line parallel to the x-axis.

        local_curves: boolean value
            (OPTIONAL) [default = True] If True the original instances are displayed as edges, otherwise if False they are displayed as dots.

        chosen_row_preds_to_plot: numpy.array of shape (1, num_samples)
            (OPTIONAL) [default = None] Returned by previous function pdp().
            Such values will be displayed as red curve in the plot.

        plot_full_curves: boolean value
            (OPTIONAL) [default = False]

        plot_object: matplotlib axes object
            (OPTIONAL) [default = None] In case the user wants to pass along his own matplotlib figure to update.
            This garantees all the possible customization.

        cell_view: boolean value
            (OPTIONAL) [default = False] It displays clusters in different cells. 
            If a list of clusters is not provided this argument is ignored.

        path: string
            (OPTIONAL) [default = None] Provide here the name of the file if you want to save the visualization in an image.
            If an empty string is given, the name of the file is automatically computed.

        """

        fix = self.the_feature
        class_array = self.cls_arr
        model = self.mdl
        df_test = self.df


        num_samples = self.n_smpl


        data_set_pred_index = self.data_set_pred_index


        def plotting_prediction_changes( single_curve_object, 
                                        color_curve = "blue",
                                        spag = False, 
                                        pred_spag = None, 
                                        chosen_row_preds = None, 
                                        full_curves = False,
                                        plot_object = None,
                                        ticks_color = "black"):



            dist_matrix = single_curve_object.get_dm()
            pred_matrix = single_curve_object.get_preds()
            rWarped = single_curve_object.get_keogh_radius()
            indices_from_test = single_curve_object.get_ixs()

            if plot_object is not None:
                ax = plot_object

            else:
                fig, ax = plt.subplots(figsize=(16, 9), dpi=100)





            def b_spline(x,y):
                n_local_points = len(x)
                t = range(n_local_points)
                ipl_t = np.linspace(0.0, n_local_points - 1, 100)

                x_tup = si.splrep(t, x, k=3)
                y_tup = si.splrep(t, y, k=3)

                x_list = list(x_tup)
                xl = x.tolist()
                size_seros = len(x_list[1]) - len(xl)
                x_list[1] = xl + np.zeros(size_seros).tolist()

                y_list = list(y_tup)
                yl = y.tolist()
                size_seros = len(y_list[1]) - len(yl)
                y_list[1] = yl + np.zeros(size_seros).tolist()

                x_i = si.splev(ipl_t, x_list)
                y_i = si.splev(ipl_t, y_list)
                return x_i, y_i

            dictLabtoIndex = self.dictLabtoIndex
            original_preds = self.original_preds
            changing_rows = self.changing_rows
            df_sample = self.df_sample
            df_features = self.df_features

            trasparenza = 0.1
            dot_size = 5

            #cmap = plt.get_cmap("gist_rainbow")
            #cmap = plt.get_cmap("RdYlBu")
            # http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=10

            original_data_sample = self._back_to_the_original(list(df_sample[fix]), fix)


            #t = time.time()
            num_rows = len(pred_matrix)

            if single_cluter and plot_object is None:
                ax.set_title("1D partial dependency of " + fix + " for cluster " + str(label_title_cluster), fontsize=font_size_par)
            elif cell_view and multi_clusters:
                if clust_number <= 15:
                    ax.set_title("Cluster " + str(label_title_cluster) + " - size: "+label2.split(" : ")[1], fontsize=font_size_par)
                else:
                    ax.set_title("")
            else:
                ax.set_title("1D partial dependency of " + fix, fontsize=font_size_par)

            if full_curves:
                for i in range(num_rows):
                    ax.plot(original_data_sample,pred_matrix[i],color=color_curve,alpha=trasparenza)

            if spag:
                for j in indices_from_test:

                    low_spag = originalIndex - howFarIndex
                    high_spag = originalIndex + howFarIndex + 1

                    NowSample =allSamplesOriginal[fix + "-o-" + str(j)][low_spag:high_spag]

                    NowPreds = pred_spag[j, low_spag:high_spag]

                    #plt.plot(NowSample,NowPreds,alpha=trasparenza,color=colors_10_cluster[i])
                    x_i, y_i = b_spline(np.array(NowSample), np.array(NowPreds))
                    ax.plot(x_i, y_i, alpha=0.8, color=ticks_color)

            else:
                x_point = changing_rows[indices_from_test, dictLabtoIndex[fix]]
                x_point = self._back_to_the_original(x_point, fix)
                y_point = original_preds[indices_from_test]
                ax.scatter(x_point, y_point, c=ticks_color, s=dot_size)

            if not full_curves:
                mean_preds = np.array([ np.mean(pred_matrix[:, i]) for i in range(num_samples) ])
                std_preds = np.array([ np.std(pred_matrix[:, i]) for i in range(num_samples) ])
                #plt.plot(original_data_sample,mean_preds,color="red",alpha=1)
                ax.fill_between(original_data_sample, mean_preds-std_preds, mean_preds+std_preds,color=color_curve,alpha=0.25)
            
            if chosen_row_preds is not None:
                ax.plot(original_data_sample,chosen_row_preds,color="red",lw=2)

            the_mean_value = self._back_to_the_original(df_features["mean"][fix], fix)
            if not cell_view:
                ax.axvline(x=the_mean_value, color="green", linestyle='--')
                ax.axhline(y=thresh, color="red", linestyle='--')
            ax.set_ylabel("prediction", fontsize=font_size_par)

            ax.set_xlabel(fix, fontsize=font_size_par)
            ax.set_xlim([original_data_sample[0], original_data_sample[num_samples-1]])
            ax.set_ylim([0, 1])

            if not cell_view:

                #pred = 1
                top_label = class_array[data_set_pred_index]

                ax.text(-0.09, 0.95, top_label, fontsize=font_size_par, transform=ax.transAxes)

                #pred = 0
                origin_label = class_array[1 - data_set_pred_index]

                if len(class_array) > 2:
                    origin_label = "not "+top_label

                ax.text(-0.09, 0.05, origin_label, fontsize=font_size_par, transform=ax.transAxes)



            else:
                if clust_number > 15:

                    ax.text(0.5,
                            0.5,
                            "#"+str(label_title_cluster).zfill(2),
                            fontsize=font_size_par+10,
                            transform=ax.transAxes, 
                            ha = "center",
                            va = "center",
                            alpha = 0.25)


        
        def pdp_local(fix, allSamples, chosen_row=None):
            

            list_local_sampling = self.list_local_sampling
            
            dictLabtoIndex = self.dictLabtoIndex
            rows = self.changing_rows
            num_feat = self.num_feat

            num_rows = len(rows)
            #print("changing up",num_rows,"rows")
            new_matrix_f = np.zeros((num_rows, num_samples + 1, num_feat))
            depth_index = 0
            i = 0
            for r in rows:
                sample_vals = list_local_sampling[i][fix]
                allSamples[fix + "-o-" + str(i)] = sample_vals
                #if float(i+1)%10000==0:
                    #print ("---- loading matrix: ", np.round(i/float(num_rows),decimals=2)*100,"%")
                    #print ("------ elapsed: ",int(int(time.time()-t)/60), "m")
                i+=1
                index_height = 0
                for v in sample_vals:
                    new_r = np.copy(r)
                    new_r[dictLabtoIndex[fix]] = v
                    new_matrix_f[depth_index][index_height] = new_r
                    index_height += 1
                depth_index += 1

            if chosen_row is not None:
                chosen_row_alterations = []

                for v in sample_vals:
                    arow = np.copy(chosen_row)
                    arow[dictLabtoIndex[fix]] = v
                    chosen_row_alterations.append(arow)

                return new_matrix_f, np.array(chosen_row_alterations), allSamples
            return new_matrix_f, allSamples

        def compute_pred_local(matrix_changed_rows, chosen_row_alterations=None):
            num_feat = self.num_feat
            data_set_pred_index = self.data_set_pred_index
            numS = num_samples + 1
            #t = time.time()
            num_rows= len(matrix_changed_rows)
            pred_matrix = np.zeros((num_rows, numS))
            matrix_changed_rows = matrix_changed_rows.reshape((num_rows * numS, num_feat))
            ps = model.predict_proba(matrix_changed_rows)
            ps = [ x[data_set_pred_index] for x in ps ]
            k = 0
            for i in range(0, num_rows * numS):
                if i % numS ==0:
                    pred_matrix[k] = ps[i:i + numS]
                    k += 1
            if chosen_row_alterations is not None:
                chosen_row_preds = model.predict_proba(chosen_row_alterations)
                chosen_row_preds = [ x[data_set_pred_index] for x in chosen_row_preds ]
                return pred_matrix, chosen_row_preds
            return pred_matrix



        ###########################
        # WHERE THE MAGIC HAPPENS #
        ###########################

        font_size_par = 20


        single_cluter = False
        clust_number = None
        multi_clusters = False

        if type(curves_input) == tuple:
            single_cluter = True
            label_title_cluster = curves_input[0]
            single_curve = curves_input[1]

        if type(curves_input) == list:
            multi_clusters = True
            clust_number = len(curves_input)

        if not multi_clusters and not single_cluter:
            single_curve = curves_input


        
        if plot_object is None:
            if cell_view and multi_clusters:

                grid_heigth = int(np.ceil(np.sqrt(clust_number)))
                grid_width = int(np.ceil(clust_number / grid_heigth))


                grid_heigth_real = grid_heigth
                grid_width_real = grid_width
                if grid_heigth*grid_width == clust_number:

                    grid_heigth = int(np.ceil(np.sqrt(clust_number+1)))
                    grid_width = int(np.ceil((clust_number+1) / grid_heigth))


                col_plot_init = -1
                row_plot_init = 0

                fig, ax = plt.subplots(nrows=grid_heigth, ncols=grid_width, figsize=(16, 9), dpi=100)
            else:
                fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
        else:
            ax = plot_object
            old_label = ax.get_legend_handles_labels()
            handles = old_label[0]
            labels = old_label[1]
            old_text1 = labels[::2]
            old_text2 = labels[1::2]
            old_patches1 = handles[::2]
            old_patches2 = handles[1::2]

        preds_local = None

        if local_curves:
            originalIndex = int(num_samples / 2)
            howFarIndex = int(np.ceil(2/100 * num_samples))
            howFarIndex = max([2,howFarIndex])
            allSamples = {}

            the_matrix_local, allSamples = pdp_local(fix,allSamples)
            preds_local = compute_pred_local(the_matrix_local)

            allSamplesOriginal = {}
            for key in allSamples:
                allSamplesOriginal[key] = self._back_to_the_original(allSamples[key], key.split("-o-")[0])


        texts1 = []
        texts2 = []
        color_legend = []
        end_plot = False
        if multi_clusters:
            if color_plot is None:
                color_plot_original = ['#1f78b4',
                                       '#33a02c',
                                       '#e31a1c',
                                       '#ff7f00',
                                       '#6a3d9a',
                                       '#a6cee3',
                                       '#b2df8a',
                                       '#fb9a99',
                                       '#fdbf6f',
                                       '#cab2d6']

                if len(color_plot_original) < clust_number:
                    color_plot = []
                    i_color = 0
                    for i in range(clust_number):
                        color_plot.append(color_plot_original[i_color])
                        i_color+=1
                        if i_color == len(color_plot_original) - 1:
                            i_color = 0
                else:
                    color_plot = color_plot_original


            for i in range(clust_number):


                single_curve = curves_input[i][1]
                label_title_cluster = curves_input[i][0]

                dist_matrix_this = single_curve.get_dm()
                sizeClust = len(single_curve.get_ixs())
                if sizeClust == 1:
                    avgRmse = 0
                else:
                    avgRmse = np.round(sum(sum(dist_matrix_this))/((sizeClust-1)*sizeClust),decimals=2)

                label1 = " : " + str(avgRmse)
                label2 = " : " + str(sizeClust)

                texts1.append("Clust. " + str(label_title_cluster) + label1)
                texts2.append("Clust. " + str(label_title_cluster) + label2)
                color_legend.append(color_plot[i])

                if cell_view:
                    font_size_par = 15

                    if i /  grid_heigth_real < col_plot_init + 1:
                        #print("same_row:",i /  grid_heigth,"<",col_plot_init + 1)
                        col_plot = col_plot_init 
                        row_plot = row_plot_init + 1
                    else:
                        col_plot = col_plot_init + 1
                        row_plot = 0
                    #print("r:",col_plot,"- c:",row_plot)
                    axes_plot = ax[row_plot,col_plot]

                    col_plot_init = col_plot
                    row_plot_init = row_plot

                else:
                    axes_plot = ax                    

                plotting_prediction_changes(
                            single_curve_object = single_curve, 
                            color_curve = color_plot[i],
                            ticks_color = color_plot[i],
                            spag = local_curves, 
                            pred_spag = preds_local, 
                            chosen_row_preds = chosen_row_preds_to_plot, 
                            full_curves = plot_full_curves,
                            plot_object = axes_plot)

                if cell_view:

                    if col_plot_init != 0:
                            axes_plot.set_ylabel("")
                            axes_plot.yaxis.set_ticklabels([])

                    if row_plot_init != grid_heigth_real-1 and i != clust_number-1:
                            axes_plot.set_xlabel("")
                            axes_plot.xaxis.set_ticklabels([])


        else:
            if color_plot is None:
                color_plot = "blue"
                ticks_color = "black"
            else:
                ticks_color = color_plot
            
            dist_matrix_this = single_curve.get_dm()
            sizeClust = len(single_curve.get_ixs())



            if sizeClust == 1 or dist_matrix_this is None:
                avgRmse = 0

            else:
                avgRmse = np.round(sum(sum(dist_matrix_this))/((sizeClust-1)*sizeClust),decimals=2)

            if single_cluter:
                string_legend = "Clust. " + str(label_title_cluster) + " : "
            else:
                string_legend = " : "

            label1 = string_legend + str(avgRmse)
            label2 = string_legend + str(sizeClust)

            texts1.append(label1)
            texts2.append(label2)

            color_legend.append(color_plot)

            plotting_prediction_changes(single_curve_object = single_curve, 
                                        color_curve = color_plot,
                                        ticks_color = ticks_color,
                                        spag = local_curves, 
                                        pred_spag = preds_local, 
                                        chosen_row_preds = chosen_row_preds_to_plot, 
                                        full_curves = plot_full_curves,
                                        plot_object = ax)

        if plot_object is None:
            end_plot = True


        size_legend = len(texts2)

        patches1 = [ plt.plot([],  [], marker="o", ms=10, ls="", mec=None, color=color_legend[i], 
                label="{:s}".format(texts1[i]))[0] for i in range(size_legend) ]

        patches2 = [ plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=color_legend[i], 
                label="{:s}".format(texts2[i]))[0] for i in range(size_legend) ] 
        
        pos1 = (1.01, 1)
        pos2 = (1.01, 0)



        if plot_object is not None:
            patches1 = old_patches1 + patches1
            patches2 = old_patches2 + patches2
            texts1 = old_text1 + texts1
            texts2 = old_text2 + texts2

        vals_texts_1 = [ float(v.split(" : ")[1]) for v in texts1 ]
        vals_texts_2 = [ float(v.split(" : ")[1]) for v in texts2 ]


        zipped1 = sorted(zip(vals_texts_1,texts1,patches1), reverse = False)
        zipped2 = sorted(zip(vals_texts_2,texts2,patches2), reverse = True)

        texts1 = [t[1] for t in zipped1]

        patches1 = [t[2] for t in zipped1]

        texts2 = [t[1] for t in zipped2]

        patches2 = [t[2] for t in zipped2]

        if not cell_view:

            plt.subplots_adjust(top=.9, bottom=.1, left=.075, right=.80)


            ax.legend( handles=patches1, bbox_to_anchor=pos1, 
                       loc="upper left", ncol=1, 
                       facecolor="#d3d3d3", numpoints=1, 
                       fontsize=13,frameon=False,
                       title = "Average Distance" )


            legend2 = Legend( ax, labels=texts2, handles=patches2, bbox_to_anchor=pos2,
                              loc="lower left", ncol=1, facecolor="#d3d3d3", 
                              numpoints=1, fontsize=13,frameon=False,
                              title = "# of Instances")

            for art in ax.artists:
                art.remove()

            ax.add_artist(legend2)

        elif cell_view and multi_clusters:

            ax_object_count = 0
            extra_space = False
            no_axis = False

            if grid_heigth!=grid_heigth_real:
                extra_space = True

            for j in range(grid_width):
                for i in range(grid_heigth):

                    ax_object = ax[i,j]
                    ax_object_count+=1

                    if clust_number <= 15:

                        if (ax_object_count > clust_number and not extra_space) or (extra_space and (i > grid_width_real-1 or j > grid_width_real-1)):
                            ax_object.axis("off")

                        if (ax_object_count == clust_number + 1 and not extra_space) or (extra_space and j == 0 and i == grid_heigth - 1):

                            ax_object.legend( handles=patches1, bbox_to_anchor=(0,1), 
                                loc="upper left", ncol=1, 
                                facecolor="#d3d3d3", numpoints=1, 
                                fontsize=10,frameon=False,
                                title = "Average Distance" )


                            legend2 = Legend( ax_object, labels=texts2, handles=patches2, bbox_to_anchor=(1,1),
                                  loc="upper right", ncol=1, facecolor="#d3d3d3", 
                                  numpoints=1, fontsize=10,frameon=False,
                                  title = "# of Instances")

                            for art in ax_object.artists:
                                art.remove()

                            ax_object.add_artist(legend2)
                    else:
                        no_axis = True
                        ax_object.axis("off")

            class_focus = self.cls_fcs

            title_all = 'pdp for class: '+class_focus.replace("\n"," ")
            title_all = title_all + " - feature: "+fix

            if no_axis:
                fig.subplots_adjust(wspace=0.05, hspace = 0.25, top=0.9,bottom=0.05,left=.05, right=.95)

            else:
                if not extra_space:
                    if clust_number == 11:
                        fig.subplots_adjust(wspace=0.05, hspace = 0.5, top=0.9,bottom=0.2,left=.05, right=.95)
                    elif clust_number > 11 and clust_number < 15:
                        fig.subplots_adjust(wspace=0.05, hspace = 0.5, top=0.9,bottom=0.08,left=.05, right=.95)
                    elif clust_number == 15:
                        fig.subplots_adjust(wspace=0.05, hspace = 0.5, top=0.9,bottom=0.3,left=.05, right=.95)
                    else:
                        fig.subplots_adjust(wspace=0.05, hspace = 0.5, top=0.9,bottom=0.1,left=.05, right=.95)
                else:
                    if clust_number > 15:
                        # 16 25 ...
                        fig.subplots_adjust(wspace=0.05, hspace = 0.5, top=0.9,bottom=0.12,left=.05, right=.95)
                    elif clust_number >= 9 and clust_number < 15:
                        #9
                        fig.subplots_adjust(wspace=0.05, hspace = 0.5, top=0.9,bottom=0.12,left=.05, right=.95)
                    else:
                        #4
                        fig.subplots_adjust(wspace=0.05, hspace = 0.5, top=0.9,bottom=-0.05,left=.05, right=.95)

            fig.suptitle(title_all, fontsize=font_size_par)



        # no sides
        # plt.tight_layout()
        # only right side
        #plt.subplots_adjust(top=.9, bottom=.1, left=.05, right=.80)
        # both sides
        
        
        # only left side
        #plt.subplots_adjust(top=.9, bottom=.1, left=.085, right=.99)
        # does not work: ax.margins(y=0.05)



        if end_plot:

            if path is not None:
                if len(path) == 0 or type(path) is not str:
                    path = "plot_" + fix + ".png"

                fig.savefig(path)
            plt.show()
            plt.close("all")


    def _local_sampling_mult (self, fix, base_value, based_on_mean = False, min_max = None):

        
        '''
        This private function is needed to perform sampling based on an arbitrary set of instances.

        Parameters
        ----------
        
        fix : string
            (REQUIRED) The feature name.

        base_value : float
            (REQUIRED) The value which will be placed in the center of the sampling,
            needed to define the center of the heatmap.

        based_on_mean : boolean
            (OPTIONAL) [default = False] If True the sampling is around the mean value of the set of istances for the chosen feature.
            The distance around such value is fixed and depends on the standard deviation.
            If False the sampling goes to min to max of the value of the set of istances for the chosen feature.
            Therefore base_value should be set at (max-min)/2 + min

        min_max: tuple of two floats
            (NOT REQUESTED  if based_on_mean) If we are sampling around the mean this argument is ignored.
            (REQUESTED      if not based_on_mean) If instead we are sampling from min to max this parameter is needed.

        Returns
        -------

        final_samples: list of floats of size num_samples
            the samples of the feature to be used to compute the pdp.



       '''

        num_samples = self.n_smpl


        if based_on_mean:


            samplesLeft = list( np.linspace( base_value - 1, 
                                             base_value, 
                                             int(num_samples / 2) + 1 ) )

            samplesRight = list( np.linspace( base_value, 
                                              base_value + 1, 
                                              int(num_samples / 2) + 1) )

            samples = samplesLeft + samplesRight
            
            divisor = int(num_samples/2+1)
            final_samples = samples[:divisor-1]+[base_value]+samples[divisor+1:]

        else:

            if min_max is not None:

                min_val = min_max[0]
                max_val = min_max[1]

                samplesLeft = list( np.linspace( min_val, 
                                                 base_value, 
                                                 int(num_samples / 2) + 1 ) )

                samplesRight = list( np.linspace( base_value, 
                                                  max_val, 
                                                  int(num_samples / 2) + 1) )

                samples = samplesLeft + samplesRight
                
                divisor = int(num_samples/2+1)
                final_samples = samples[:divisor-1]+[base_value]+samples[divisor+1:]

            else:
                print("Error: max and min tuple not provided")

        return final_samples


    def pdp_2D(self, A, B, instances = None, sample_data = None, zoom_on_mean = False):

        """
        By choosing two features and changing them in two sample ranges, 
        for each provided instance we can create num_samples X num_samples different versions of the original instance.
        Then we are able to compute prediction values for each of the different vectors.
        pdp_2D() initialize and returns a python object from the class Pdp2DCurves() containing such predictions values. 

        Parameters
        ----------

        A : string
            (REQUIRED) The name of a feature as reported in one of the df_test columns.

        B : string
            (REQUIRED) The name of another feature as reported in one of the df_test columns.
       
        instances : list of integers OR single integer value
            (OPTIONAL) [default = None] If provided, it needs to contain the indexes of the test-set chosen instances.
            If None all indexes in test set are used. (range(lenTest))

        sample_data: tuple (list for sample A, list for sample B)
            (OPTIONAL) [default = None] The sample values for feature A and feature B can be provided as arguments here.
            If not provided they will be automatically computed with the private funct. _local_sampling_mult() depending on zoom_on_mean.

        zoom_on_mean: boolean
            (OPTIONAL) [default = False] If sample_data is not None, then this argument is ignored.
            Otherwise:   if True sampling is around the value (mean value feature A, mean value feature B) of the set of instances.
                         if False sampling goes from min to max of the two feature values, 
                         therefore all outliers will be part of the sample ranges.

        Returns
        -------

        heatmap_curves: python object
            (ALWAYS) An itialized object from the class Pdp2DCurves.
            It contains all the predictions obtained from the different versions of the instances,
            along with other useful information to keep store for later steps.


        """

        rows = self.changing_rows
        dictLabtoIndex = self.dictLabtoIndex
        num_feat = self.num_feat
        num_samples = self.n_smpl
        model = self.mdl
        data_set_pred_index = self.data_set_pred_index
        original_preds = self.original_preds
        lenTest = self.lenTest
        list_local_sampling = self.list_local_sampling



        def get_data_2d(instance,samples_A,samples_B):

            def compute_pred_2d(matrix_2d):

                num_rows= len(matrix_2d)
                pred_matrix = np.zeros((num_rows,(num_samples+1)))

                matrix_2d = matrix_2d.reshape((num_rows*(num_samples+1), num_feat))

                ps = model.predict_proba(matrix_2d)

                ps = [x[data_set_pred_index] for x in ps]

                k = 0

                for i in range(0,num_rows*(num_samples+1)):
                    if i%(num_samples+1) ==0:
                        pred_matrix[k] = ps[i:i+(num_samples+1)]
                        k+=1


                return pred_matrix

            matrix_to_predict = np.zeros((num_samples+1,num_samples+1,num_feat))
            row_instance = rows[instance]
            depthIndex = 0

            for a in sampleA:

                indexHeight = 0

                for b in sampleB:

                    newVector = np.copy(row_instance)
                    newVector[dictLabtoIndex[A]] = a
                    newVector[dictLabtoIndex[B]] = b
                    matrix_to_predict[depthIndex][indexHeight] = newVector
                    indexHeight+=1

                depthIndex+=1

            preds = compute_pred_2d(matrix_to_predict)
            original_pred = original_preds[instance]
            return preds, original_pred

        single = False
        mult = False

        if instances is None:
            instances = list(range(lenTest))
            mult = True
        else:
            if type(instances) is int:
                single = True

            elif type(instances) is list:

                if len(instances) == 1:
                    instances = instances[0]
                    single = True

                elif len(instances) > 1:
                    mult = True

                else:
                    print("instances arg. is not suitable")
                    return

            else:
                print("instances arg. is not suitable")
                return

        if sample_data is None:

            if mult:

                if zoom_on_mean:

                    A_center = np.mean(rows[instances,dictLabtoIndex[A]])
                    B_center = np.mean(rows[instances,dictLabtoIndex[B]])

                    sampleA = self._local_sampling_mult(A, A_center, based_on_mean = True)
                    sampleB = self._local_sampling_mult(B, B_center, based_on_mean = True)

                else:

                    max_value_A = max(rows[instances,dictLabtoIndex[A]])
                    min_value_A = min(rows[instances,dictLabtoIndex[A]])
                    tuple_bounds_A = (min_value_A,max_value_A)

                    max_value_B = max(rows[instances,dictLabtoIndex[B]])
                    min_value_B = min(rows[instances,dictLabtoIndex[B]])
                    tuple_bounds_B = (min_value_B,max_value_B)

                    A_center =  min_value_A + (tuple_bounds_A[1] - tuple_bounds_A[0]) / 2
                    B_center =  min_value_B + (tuple_bounds_B[1] - tuple_bounds_B[0]) / 2

                    sampleA = self._local_sampling_mult(A, A_center, min_max = tuple_bounds_A)
                    sampleB = self._local_sampling_mult(B, B_center, min_max = tuple_bounds_B)

            else:

                sampleA = list_local_sampling[instances][A]
                sampleB = list_local_sampling[instances][B]

        else:


            sampleA = sample_data[0]
            sampleB = sample_data[1]



        if mult:

            orginal_preds = []
            raw_preds_all = []
            orig_pred_list = []

            for i in instances:
                
                raw_preds, orig_pred = get_data_2d(i,sampleA,sampleB)
                
                raw_preds_all.append(raw_preds)
                orig_pred_list.append(orig_pred)
                

            raw_preds_all = np.array(raw_preds_all)
            orig_pred_list = np.array(orig_pred_list)

        else:

            raw_preds, orig_pred = get_data_2d(instances, sampleA, sampleB)
            raw_preds_all = np.array([raw_preds])
            orig_pred_list = np.array([orig_pred])
            instances = [instances]

        heatmap_curves = Pdp2DCurves(raw_preds_all, orig_pred_list, np.array(instances), A, B)

        if mult:

            heatmap_curves.set_sample_data(sampleA,sampleB)

        ###############################################

        return heatmap_curves


    def plot_heatmap( self, curves_objs, path = None, for_splom = False, plot_object = None): 

        '''
        This function is able to plot heatmaps from python objects of the class Pdp2DCurves().

        Parameters
        ----------

        curves_objs : python object
            (REQUIRED) A python object from the class Pdp2DCurves(). ( Returned by previous function pdp_2D() )
           Otherwise a list of such python objects in tuples. ( Returned by previous function compute_clusters() )
           In the latter case the visualization will have an heatmap for each cluster in the list.

        path: string
            (OPTIONAL) [default = None] Provide here the name of the file if you want to save the visualization in an image.
            If an empty string is given, the name of the file is automatically generated.

        for_splom: boolean
            (OPTIONAL) [default = False] This function is later used to generate the cell heatmaps in a SPLOM visualization.
            This flag is passed True when such call is needed. In all other cases this flag is False.
        
        plot_object: matplotlib axes object
            (OPTIONAL) [default = None] In case the user wants to pass along his own matplotlib figure to update.
            This garantees all the possible customization.

        '''       

        rows = self.changing_rows
        dictLabtoIndex = self.dictLabtoIndex
        num_samples = self.n_smpl
        num_feat = self.num_feat
        list_local_sampling = self.list_local_sampling

        def single_heatmap (curves_input,  plot_obj, for_splom = for_splom, clustering = False):

            if type(curves_input) is tuple:
                label_cluster = curves_input[0]
                curves_input = curves_input[1]

            heatmap_data = curves_input.get_data()
            instances = list(curves_input.get_ixs())
            size = str(len(instances))
            feat_x = curves_input.get_B()
            feat_y = curves_input.get_A()

            single = False
            mult = False
            font_title = font_size_par


            if type(instances) is int:
                single = True

            elif type(instances) is list:

                if len(instances) == 1:
                    instances = instances[0]
                    single = True


                elif len(instances) > 1:
                    mult = True

                else:
                    print("instances arg. is not suitable")
                    return

            else:
                print("instances arg. is not suitable")
                return


            color_data = heatmap_data.reshape((-1,))
            color_parameter = max( [ max(color_data),
                                     abs(min(color_data)) ] )


            if for_splom or clustering:
            #se mult
                color_parameter = 0.5


            axis = plot_obj

            
            if clustering:
                string_clust = "Cluster "
                if clust_number > 12:
                    string_clust ="#"
                    font_title -=2
                    if clust_number > 30:
                        font_title -=2
                axis.set_title(string_clust+str(label_cluster).zfill(2)+ " - size: "+size, fontsize=font_title)


            axis.set_xlabel(feat_x,fontsize= font_title)
            axis.set_ylabel(feat_y,fontsize= font_title)
            

            if mult:
            
                A_points = self._back_to_the_original(rows[instances,dictLabtoIndex[feat_y]], feat_y)
                B_points = self._back_to_the_original(rows[instances,dictLabtoIndex[feat_x]], feat_x)


            color_scale = mpl.colors.Normalize(vmin=-np.abs(color_parameter),vmax=np.abs(color_parameter))
            

            if single and not clustering:
                sampleA = list_local_sampling[instances][feat_y]
                sampleB = list_local_sampling[instances][feat_x]

            else:

                sampleA, sampleB = curves_input.get_sample_data()
            
            div_index = int(num_samples/2)
            A_center = sampleA[div_index]
            B_center = sampleB[div_index]



            sampleA = self._back_to_the_original(list(sampleA),feat_y)
            sampleB = self._back_to_the_original(list(sampleB),feat_x)

            A_center = self._back_to_the_original(A_center,feat_y)
            B_center = self._back_to_the_original(B_center,feat_x)




            def ticksCreator(sample,num_ticks):
                
                #skip_pase = int(np.ceil(num_samples/num_ticks))
                #sample = list(sample)[::skip_pase]
                sample = list(sample)

                base_value = sample[int(num_samples/2)]
                min_val = sample[0]
                max_val = sample[-1]

                samplesLeft = list( np.linspace( min_val, 
                                                 base_value, 
                                                 int(num_ticks / 2) + 1 ) )

                samplesRight = list( np.linspace( base_value, 
                                                  max_val, 
                                                  int(num_ticks / 2) + 1) )

                samples = samplesLeft + samplesRight
                divisor = int(num_ticks/2+1)

                sample = samples[:divisor-1]+[base_value]+samples[divisor+1:]



                return sample, min_val, max_val 

            # num_ticks has to be an even integer number
            if not clustering:
                num_ticks = min(num_samples,20)
            else:
                num_ticks = min(num_samples,20)
                if clust_number > 2:
                    num_ticks = min(num_samples,10)
                    if clust_number > 12:
                        num_ticks = min(num_samples,4)


            ticksA, minA, maxA = ticksCreator(sampleA,num_ticks)
            ticksB, minB, maxB = ticksCreator(sampleB,num_ticks)




            ext = [minB,maxB,maxA,minA]
            heat = axis.imshow( heatmap_data, 
                                cmap="RdYlBu", norm = color_scale, 
                                extent=ext, aspect = "auto")

            if not for_splom and not clustering:
                divider = make_axes_locatable(axis)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(heat,cax=cax)
            



            axis.set_yticks(ticksA, minor=False)
            axis.set_xticks(ticksB, minor=False)

            if clustering:
                axis.set_yticklabels([str(np.around(t,decimals=3)) for t in ticksA], minor=False)
                axis.set_xticklabels([str(np.around(t,decimals=3)) for t in ticksB], minor=False)


            axis.set_xlim(minB, maxB)
            axis.set_ylim(maxA, minA)


            size_marker = 50
            edge_marker = 3
            marker_style = "+"

            if for_splom:

                size_marker = size_marker / num_feat
                edge_marker = edge_marker / num_feat

            elif not clustering:

                axis.plot(B_center, A_center,
                          marker=marker_style,
                          markersize=size_marker,
                          color="black", 
                          markeredgewidth=edge_marker)

            if mult:

                if clustering:

                    marker_style_small = "."
                    edge_marker_small = 0
                    size_marker_small = 40 / max(1,np.ceil(max(grid_heigth,grid_width)/3))
                    if len(instances) > 10:
                        opacity_small = 0.25
                    else:
                        opacity_small = 1

                else:

                    if len(instances) > 100:
                        marker_style_small = "."
                        size_marker_small = 40
                        edge_marker_small = 0
                        opacity_small = 0.5

                        if for_splom:
                            size_marker_small = size_marker_small / max(1,np.ceil(num_feat/3))
                            opacity_small = 0.05


                    else:
                        marker_style_small = "+"
                        size_marker_small = 50
                        edge_marker_small = 2

                        if len(instances) > 10:
                            opacity_small = 0.25
                        else:
                            opacity_small = 1


                        if for_splom:
                            marker_style_small = "."
                            edge_marker_small = 0
                            size_marker_small = 40 / max(1,np.ceil(num_feat/3))


                axis.scatter(B_points, A_points, 
                             marker = marker_style_small, 
                             s = size_marker_small,
                             linewidths = edge_marker_small, 
                             color = "black",
                             alpha = opacity_small)

            middle_tick = int(num_ticks/2)

            axis.get_xticklabels()[middle_tick].set_color('red') 
            axis.get_yticklabels()[middle_tick].set_color('red')
            
            for tick in axis.get_xticklabels():
                tick.set_rotation(45)

            if clustering:
                if row_plot != grid_heigth -1:
                    axis.set_xlabel("",fontsize= font_size_par)
                if col_plot != 0:
                    axis.set_ylabel("",fontsize= font_size_par)




        clust_flag = False
        if type(curves_objs) is list:
            clust_flag = True
            clust_number = len(curves_objs)

            if plot_object is not None:

                axis = plot_object

            else:

                grid_heigth = int(np.ceil(np.sqrt(clust_number)))
                grid_width = int(np.ceil(clust_number / grid_heigth))


                col_plot_init = -1
                row_plot_init = 0

                fig, ax = plt.subplots( figsize=(10.8,10.8), nrows=grid_heigth, ncols=grid_width, dpi=100)


                for i in range(clust_number):

                    if i /  grid_heigth < col_plot_init + 1:

                        col_plot = col_plot_init 
                        row_plot = row_plot_init + 1

                    else:
                        col_plot = col_plot_init + 1
                        row_plot = 0

                    if grid_width == 1 :
                        axes_plot = ax[row_plot]
                    else:
                        axes_plot = ax[row_plot,col_plot]

                    col_plot_init = col_plot
                    row_plot_init = row_plot
                    
                    font_size_par = 15

                    single_heatmap (curves_objs[i], plot_obj = axes_plot, clustering = True)
                    if clust_number > 16:
                        axes_plot.axis("off")

                while True:
                    try:
                        row_plot_init +=1
                        ax[row_plot_init,col_plot_init].axis("off")
                    except IndexError:
                        break




        else:
            if plot_object is None: 
                fig, ax = plt.subplots( figsize=(10.8,10.8), dpi=100)
            else:
                ax = plot_object

            font_size_par = 20
            single_heatmap (curves_objs, plot_obj = ax)

            if not for_splom:

                fig.subplots_adjust( top = 0.95, 
                                     bottom = 0.05, 
                                     left = 0.05, 
                                     right = 0.95 )


        if plot_object is None:

            if clust_flag:
                color_parameter = 0.5

                color_scale = mpl.colors.Normalize(vmin = -np.abs(color_parameter), vmax = np.abs(color_parameter))

                cax = fig.add_axes([0.9, 0.1, 0.05, 0.8])
                cb1 = mpl.colorbar.ColorbarBase(cax, cmap = "RdYlBu", norm = color_scale)

                w_s = 0.2
                h_s = 0.55
                bot = 0.1
                top = 0.9
                lef = 0.1
                rig = 0.85

                if clust_number > 6:
                    w_s += 0.05
                    if clust_number > 9:
                        h_s += 0.1
                        if clust_number > 12:
                            w_s += 0.1



                fig.subplots_adjust( wspace = w_s, 
                                     hspace = h_s, 
                                     top = top, 
                                     bottom = bot, 
                                     left = lef, 
                                     right = rig)

            if not for_splom:
                class_focus = self.cls_fcs
                title_all = '2d pdp for class: '+class_focus.replace("\n"," ")
                fig.suptitle(title_all, fontsize=font_size_par)

            if path is not None:
                if len(path) == 0:
                    path = "heat_map.png"
                fig.savefig(path)

            plt.show()
            plt.close("all")


    



    def get_data_splom(self, instances_input = None, zoom_on_mean = False):

        '''
        Creates the data needed for a SPLOM visualization starting from a list of indexes relative to the desired instances to visualize:

        Parameters
        ----------

        instances_input : list of integers OR single integer value
            (OPTIONAL) [default = None] If provided, it needs to contain the indexes of the test-set chosen instances.
            If None all indexes in test set are used. (range(lenTest))

        zoom_on_mean: boolean
            (OPTIONAL) [default = False] If True sampling of each feature is around its mean value from the set instances_input.
                         if False sampling goes from min to max of each feature, 
                         therefore all outliers will be part of the sample ranges.

        Returns
        -------

        cell_object_dict: python dictionary
            (ALWAYS) grid position heatmap (key) : python object from class Pdp2DCurves() for heatmap (value)

        '''

        num_feat = self.num_feat

        dictLabtoIndex = self.dictLabtoIndex

        lenTest = self.lenTest

        rows = self.changing_rows

        list_local_sampling = self.list_local_sampling


        single = False
        mult = False

        if instances_input is None:
            instances_input = list(range(lenTest))
            mult = True
        else:
            if type(instances_input) is int:
                single = True

            elif type(instances_input) is list:

                if len(instances_input) == 1:
                    instances_input = instances[0]
                    single = True

                elif len(instances_input) > 1:
                    mult = True

                else:
                    print("instances arg. is not suitable")
                    return

            else:
                print("instances arg. is not suitable")
                return

        list_feat = list(dictLabtoIndex.keys())

        pairs_of_features = []
        for comb in combinations(list(range(num_feat)), 2):
            pairs_of_features.append(comb)

        df_sample_splom = pd.DataFrame()

        for feat in list_feat:

            if mult:

                if zoom_on_mean:

                    center = np.mean(rows[instances_input,dictLabtoIndex[feat]])

                    sample = self._local_sampling_mult(feat, center, based_on_mean = True)

                else:

                    max_value = max(rows[instances_input,dictLabtoIndex[feat]])
                    min_value = min(rows[instances_input,dictLabtoIndex[feat]])
                    tuple_bounds = (min_value, max_value)

                    center =  min_value + (tuple_bounds[1] - tuple_bounds[0]) / 2

                    sample = self._local_sampling_mult(feat, center, min_max = tuple_bounds)

            else:

                sample = list_local_sampling[instances_input][feat]


            df_sample_splom[feat] = sample

        cell_object_dict = {}

        for ij in pairs_of_features:
            i = ij[0]
            j = ij[1]
            feat_A = list_feat[i]
            feat_B = list_feat[j]

            A_sample = df_sample_splom[feat_A]
            B_sample = df_sample_splom[feat_B]


            splom_cell = self.pdp_2D( feat_A, 
                                      feat_B, 
                                      instances_input, 
                                      sample_data = (A_sample, B_sample), 
                                      zoom_on_mean = zoom_on_mean)


            cell_object_dict[ij] = splom_cell

        return cell_object_dict




    def plot_splom(self, heatmaps_objects, path = None):

        '''
        visualizes a SPLOM of heatmaps that show every possible combination pair of features.
        This visualization take quite some time, especially when num_feat is great.

        Parameters
        ----------

        heatmaps_objects: python dictionary or list of dictionaries.
            (REQUIRED) Data for visualization returned by get_data_splom().
            If it is a list a splom will be visualized for each cluster.

        path: string
            (OPTIONAL) [default = None] Provide here the name of the file if you want to save the visualization in an image.
            If an empty string is given, the name of the file is automatically generated.

        '''

        num_feat = self.num_feat

        dictLabtoIndex = self.dictLabtoIndex

        lenTest = self.lenTest

        list_feat = list(dictLabtoIndex.keys())



        clust = False
        if type(heatmaps_objects) is list:
            clust = True
            clust_number = len(heatmaps_objects)
        label_cluster = ""
        if type(heatmaps_objects) is tuple:
            label_cluster = "Cluster #"+str(heatmaps_objects[0]).zfill(2)+" - "
            heatmaps_objects = heatmaps_objects[1]


        if clust:
            fig = plt.figure(figsize=(16,9), dpi=100)
            grid_heigth = int(np.ceil(np.sqrt(clust_number)))
            grid_width = int(np.ceil(clust_number / grid_heigth))
            outer = gridspec.GridSpec(grid_width, grid_heigth, wspace=0.1, hspace=0.1)
   
        else:
            fig, axes = plt.subplots(nrows=num_feat, ncols=num_feat, figsize=(10.8,10.8), dpi=100)

        pairs_of_features = []
        for comb in combinations(list(range(num_feat)), 2):
            pairs_of_features.append(comb)

        if clust:
            from_grid_to_index = {}
            actual_grid_size = num_feat+1
            for h in range(int(actual_grid_size**2)):
                i = int(np.floor(h / actual_grid_size))
                j = h - i*actual_grid_size
                from_grid_to_index[(i,j)] = h


        if not clust:

            for ij in pairs_of_features:
                i = ij[0]
                j = ij[1]
                splom_cell = heatmaps_objects[ij]
                self.plot_heatmap(splom_cell, plot_object=axes[i,j], for_splom = True)
                axes[i,j].axis("off")
                swapped_splom_cell = splom_cell.swap_features()
                self.plot_heatmap(swapped_splom_cell, plot_object=axes[j,i], for_splom = True)
                axes[j,i].axis("off")


            for d in range(num_feat):
                axes[d,d].annotate( list_feat[d].replace(" ","\n"), 
                        (0.5, 0.5), 
                        xycoords='axes fraction',
                        ha='center', va='center')
                axes[d,d].axis("off")



        else:
            label_size = 15
            if clust_number > 9:
                label_size-=5
            for clstr in range(clust_number):

                inner = gridspec.GridSpecFromSubplotSpec(num_feat+1, num_feat+1, subplot_spec = outer[clstr], wspace = 0.15, hspace = 0.15)
                heatmaps_objects_cluster = heatmaps_objects[clstr][1]
                #label_cluster = "#"+str(heatmaps_objects[clstr][0]).zfill(2)

                i_cl = int(np.floor(clstr / grid_heigth))
                j_cl = clstr - i_cl*grid_heigth
                display_top = False
                display_lef= False
                if i_cl == 0:
                    display_top = True
                if j_cl == 0:
                    display_lef = True

                for ij in pairs_of_features:
                    i = ij[0]+1
                    j = ij[1]+1
                    h_up = from_grid_to_index[(i,j)]
                    h_down = from_grid_to_index[(j,i)]


                    ax = plt.Subplot(fig, inner[h_up])

                    splom_cell = heatmaps_objects_cluster[ij]
                    self.plot_heatmap(splom_cell, plot_object=ax, for_splom = True)
                    ax.axis("off")
                    fig.add_subplot(ax)

                    ax = plt.Subplot(fig, inner[h_down])
                    swapped_splom_cell = splom_cell.swap_features()
                    self.plot_heatmap(swapped_splom_cell, plot_object=ax, for_splom = True)
                    ax.axis("off")
                    fig.add_subplot(ax)


                for f in range(num_feat):

                    h_label_top = from_grid_to_index[(0,f+1)]
                    h_label_left = from_grid_to_index[(f+1,0)]

                    feat = list_feat[f][:5]+"."
                    if display_top:
                        ax = plt.Subplot(fig, inner[h_label_top])
                        ax.annotate(feat, 
                                    (0.4, 0), 
                                    xycoords='axes fraction',
                                    ha='left', va='bottom',
                                    size = label_size, rotation = 45)
                        ax.axis("off")
                        fig.add_subplot(ax)
                    if display_lef:
                        ax = plt.Subplot(fig, inner[h_label_left])
                        ax.annotate(feat, 
                                    (1.0, 0.5), 
                                    xycoords='axes fraction',
                                    ha='right', va='center',
                                    size = label_size, rotation = 0)
                        ax.axis("off")                    
                        fig.add_subplot(ax)



        color_parameter = 0.5

        color_scale = mpl.colors.Normalize(vmin = -np.abs(color_parameter), vmax = np.abs(color_parameter))

        cax = fig.add_axes([0.85, 0.05, 0.05, 0.9])
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap = "RdYlBu", norm = color_scale)
        if clust:
            fig.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.8)
        else:
            fig.subplots_adjust(wspace = 0.15, hspace = 0.15, top=0.9, bottom=0.05, left=0.05, right=0.8)


        font_size_par = 20
        class_focus = self.cls_fcs

        title_all = label_cluster+'pdp for class: '+class_focus.replace("\n"," ")
        fig.suptitle(title_all, fontsize=font_size_par)


        if path is not None:
            if len(path) == 0:
                path = "splom.png"
                if clust:
                    path = "splom_clustering.png"
            fig.savefig(path)

        plt.show()
        plt.close("all")
