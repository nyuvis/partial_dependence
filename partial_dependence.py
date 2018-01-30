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

    
__version__ = "0.0.2"

class PdpCurves(object):


    def __init__(self, preds):
        self._preds = preds
        self._ixs = np.arange(self._preds.shape[0])

        self._dm = None
        self.r_param = None

        #self.labels_cluster = None


    def copy(self):
        return self.__copy__()

    def __copy__(self):
      res = type(self)(self._preds)
      res.__dict__.update(self.__dict__)
      return res

    def split(self,labels_cluster):
        if labels_cluster is None:
            return [ (None, self.copy()) ]

        def get_slice(c, lbl):
            c._preds = self._preds[labels_cluster == lbl, :]
            if self._dm is not None:
                c._dm = self._dm[labels_cluster == lbl, :][:, labels_cluster == lbl]
            c._ixs = self._ixs[labels_cluster == lbl]
            return c

        list_of_clusters = np.unique(labels_cluster)
        return [ (lbl, get_slice(self.copy(), lbl)) for lbl in list_of_clusters ]
    def get_ixs(self):
        return self._ixs

    #def write_labels(self,labels_array):
        #self.labels_cluster = labels_array

    #def get_labels(self):
        #return self.labels_cluster

    def get_preds(self):
        return self._preds

    def get_dm(self):
        if self._dm is None:
            self._dm = self._compute_dm()
        return self._dm

    def _compute_dm(self):
        preds = self._preds
        lenTest = len(self._preds)

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

        #self.lb_keogh_bool = lb_keogh_bool

        list_of_test_indexes = range(lenTest)
        pairs_of_curves = []
        for comb in combinations(list_of_test_indexes, 2):
            pairs_of_curves.append(comb)

        k = 0
        all_total = len(pairs_of_curves)
        distance_matrix = np.zeros((lenTest, lenTest))
        #start_time = time.time()
        for pair in pairs_of_curves:
            i = pair[0]
            j = pair[1]

            k+=1
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
        self.n_smpl = num_samples
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

        eps = 0.01
        for laballa in xlabel:
            lower_bound = df_features["min"][laballa] - eps
            higher_bound = df_features["max"][laballa] + eps
            #bound = df_features["mean"][laballa] + 2*df_features["sd"][laballa]
            df_sample[laballa] = np.linspace(lower_bound, higher_bound, num_samples)

        changing_rows = np.copy(x_array)
        
        self.changing_rows = changing_rows
        self.dictLabtoIndex = dictLabtoIndex
        self.original_preds = original_preds
        self.num_feat = num_feat
        self.lenTest = lenTest
        self.data_set_pred_index = data_set_pred_index
        self.df_sample = df_sample
        self.df_features = df_features
        self.de_norm_bool = de_norm_bool
        
    def pdp(self, fix, chosen_row=None,batch_size=0):

        """
        Produces for each instance the test-set num_samples different versions.
        The versions vary just for the feature fix, which changes within the sample df_sample[fix].
        All the other features values remain the same.

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
            
        #t = time.time()
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
                if number_all_preds_in_batch < num_samples:
                    print ("Error: batch size cannot be less than sample size.")
                    return np.nan
                #t = time.time()
                num_feat = self.num_feat
                data_set_pred_index = self.data_set_pred_index

                num_rows= len(matrix_changed_rows)
                pred_matrix = np.zeros((num_rows, num_samples))

                #number_all_preds_in_batch = 1000

                num_of_instances_in_batch = int( np.floor( number_all_preds_in_batch / num_samples ) )
                how_many_calls = int( np.ceil( num_rows / num_of_instances_in_batch ) )
                residual = num_of_instances_in_batch * how_many_calls - num_rows
                num_of_instances_in_last_batch = num_of_instances_in_batch - residual

                #print ("num_of_instances_in_batch"  ,  num_of_instances_in_batch)
                #print ( "how_many_calls" , how_many_calls )
                #print ( "residual" , residual )
                #print ( "num_of_instances_in_last_batch" , num_of_instances_in_last_batch )


                for i in range(0, how_many_calls):
                    #if float(i+1)%1000==0:
                        #print ("---- loading preds: ", np.round(i/float(num_rows),decimals=4)*100,"%")
                        #print ("------ elapsed: ",int(int(time.time()-t)/60), "m")


                    low_bound_index = i*num_of_instances_in_batch
                    high_bound_index = low_bound_index + num_of_instances_in_batch

                    if i == how_many_calls -1 and residual != 0:
                        high_bound_index = low_bound_index + num_of_instances_in_last_batch

                    matrix_batch = matrix_changed_rows[low_bound_index:high_bound_index]

                    pred_matrix_batch = compute_pred(self,matrix_batch)

                    pred_matrix[low_bound_index:high_bound_index] = np.copy(pred_matrix_batch)

                curves_returned = PdpCurves(pred_matrix)

                if chosen_row_alterations_sub_funct is not None:
                    chosen_row_preds = model.predict_proba(chosen_row_alterations_sub_funct)
                    chosen_row_preds = np.array([x[data_set_pred_index] for x in chosen_row_preds])
                    return curves_returned, chosen_row_preds
                return curves_returned

            if batch_size_input != 0:
                return compute_pred_in_chunks(self, matrix_changed_rows, number_all_preds_in_batch = batch_size_input, 
                    chosen_row_alterations_sub_funct = chosen_row_alterations_input)
            return compute_pred(self, matrix_changed_rows, chosen_row_alterations_sub_funct = chosen_row_alterations_input)


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
            Compute the optimal value for the parameter needed to compute the LB Keogh distance.
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
            self.rWarpedUsed = rWarpedUsed

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
            (REQUIRED) Returned by previous function pred_comp_all().

        clust_number : integer value
            (OPTIONAL) [default = 5] The number of desired clusters.

        """

        distance_matrix = curves.get_dm()



        #print()
        #print("elapsed: ",np.around( (time.time() - start_time) / 60.0, decimals = 2),"m")
        #clust = AgglomerativeClustering(affinity='precomputed', n_clusters=clust_number, linkage='complete')
        clust = AgglomerativeClustering(affinity='precomputed', n_clusters=clust_number, linkage='average')
        #clust = AgglomerativeClustering(affinity='euclidean', n_clusters=clust_number, linkage='ward')
        clust.fit(distance_matrix)
        #clust.fit(preds) #just if affinity='euclidean' and linkage='ward'
        self.dist_matrix = distance_matrix
        labels_array = clust.labels_

        return curves.split(labels_array)
        #curves.write_labels(labels_array)

    def plot(self,
             curves_input,
             color_plot = None,
             thresh = 0.5,
             local_curves = True,
             chosen_row_preds_to_plot = None,
             plot_full_curves = False,
             plot_object = None,
             cell_view = False):



        """
        Porduces the visualization printing it in a .png file in the current path.
        The visualization will display broad curves with color linked to different clusters.
        The orginal instances are displayed with coordinates (orginal feature value, original prediction) either as
        local curves or dots depending on the local_curves argument value.

        Parameters
        ----------
        
        curves : python object
            (REQUIRED) Returned by previous function pdp().

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

        """

        fix = self.the_feature
        class_array = self.cls_arr
        model = self.mdl
        df_test = self.df


        num_samples = self.n_smpl
        scale = self.scl
        shift = self.shft

        data_set_pred_index = self.data_set_pred_index


        def plotting_prediction_changes(
                single_curve_object, 
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

            original_data_sample = back_to_the_orginal(list(df_sample[fix]), fix)


            #t = time.time()
            num_rows = len(pred_matrix)

            if single_cluter and plot_object is None:
                ax.set_title("1D partial dependency of " + fix + " for cluster " + str(label_title_cluster), fontsize=font_size_par)
            elif cell_view and multi_clusters:
                if clust_number <= 15:
                    ax.set_title("Cluster " + str(label_title_cluster), fontsize=font_size_par)
                else:
                    ax.set_title("")
            else:
                ax.set_title("1D partial dependency of " + fix, fontsize=font_size_par)

            if full_curves:
                for i in range(num_rows):
                    ax.plot(original_data_sample,pred_matrix[i],color=color_curve,alpha=trasparenza)

            if spag:
                for j in indices_from_test:
                    NowSample =allSamplesOriginal[fix + "-o-" + str(j)][originalIndex - howFarIndex:originalIndex + howFarIndex + 1]
                    NowPreds = pred_spag[j, originalIndex - howFarIndex:originalIndex + howFarIndex + 1]
                    #plt.plot(NowSample,NowPreds,alpha=trasparenza,color=colors_10_cluster[i])
                    x_i, y_i = b_spline(np.array(NowSample), np.array(NowPreds))
                    ax.plot(x_i, y_i, alpha=0.8, color=ticks_color)

            else:
                x_point = changing_rows[indices_from_test, dictLabtoIndex[fix]]
                x_point = back_to_the_orginal(x_point, fix)
                y_point = original_preds[indices_from_test]
                ax.scatter(x_point, y_point, c=ticks_color, s=dot_size)

            if not full_curves:
                mean_preds = np.array([ np.mean(pred_matrix[:, i]) for i in range(num_samples) ])
                std_preds = np.array([ np.std(pred_matrix[:, i]) for i in range(num_samples) ])
                #plt.plot(original_data_sample,mean_preds,color="red",alpha=1)
                ax.fill_between(original_data_sample, mean_preds-std_preds, mean_preds+std_preds,color=color_curve,alpha=0.25)
            
            if chosen_row_preds is not None:
                ax.plot(original_data_sample,chosen_row_preds,color="red",lw=2)

            the_mean_value = back_to_the_orginal(df_features["mean"][fix], fix)
            if not cell_view:
                ax.axvline(x=the_mean_value, color="green", linestyle='--')
                ax.axhline(y=thresh, color="red", linestyle='--')
            ax.set_ylabel("prediction", fontsize=font_size_par)

            ax.set_xlabel(fix, fontsize=font_size_par)
            ax.set_xlim([original_data_sample[0], original_data_sample[num_samples-1]])
            ax.set_ylim([0, 1])
            if not cell_view:
                #pred = 0
                ax.text(-0.09, 0.05, class_array[1 - data_set_pred_index], fontsize=font_size_par, transform=ax.transAxes)
                #pred = 1
                ax.text(-0.09, 0.95, class_array[data_set_pred_index], fontsize=font_size_par, transform=ax.transAxes)


        
        def pdp_local(fix, allSamples, chosen_row=None):
            
            def local_sampling (fix, chosen_row, num_samples):
                base_value = chosen_row[dictLabtoIndex[fix]]
                samplesLeft = list(np.linspace(base_value-1, base_value, int(num_samples / 2) + 1))
                samplesRight = list(np.linspace(base_value, base_value + 1, int(num_samples / 2) + 1))
                samples = samplesLeft + samplesRight
                divisor = int(num_samples / 2 + 1)
                final_samples = samples[:divisor - 1] + [ base_value ] + samples[divisor + 1:]
                return final_samples
            
            dictLabtoIndex = self.dictLabtoIndex
            rows = self.changing_rows
            num_feat = self.num_feat

            num_rows = len(rows)
            #print("changing up",num_rows,"rows")
            new_matrix_f = np.zeros((num_rows, num_samples + 1, num_feat))
            depth_index = 0
            i = 0
            for r in rows:
                sample_vals = local_sampling(fix, r, num_samples)
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

        def back_to_the_orginal(data_this, fix):
            dictLabtoIndex = self.dictLabtoIndex
            de_norm = self.de_norm_bool

            if de_norm:
                integ = dictLabtoIndex[fix]
                data_that = data_this / scale[0][integ] - shift[0][integ]
            else:
                data_that = data_this
            return data_that


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
            howFarIndex = 2
            allSamples = {}

            the_matrix_local, allSamples = pdp_local(fix,allSamples)
            preds_local = compute_pred_local(the_matrix_local)

            allSamplesOriginal = {}
            for key in allSamples:
                allSamplesOriginal[key] = back_to_the_orginal(allSamples[key], key.split("-o-")[0])


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

                    if row_plot_init != grid_heigth_real-1:
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

            if grid_heigth!=grid_heigth_real  or grid_width_real!=grid_width_real:
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

            title_all = 'pdp for class: '+class_array[data_set_pred_index].replace("\n"," ")
            title_all = title_all + " - feature: "+fix

            if no_axis:
                fig.subplots_adjust(wspace=0.05, hspace = 0.25,top=0.9,bottom=0.05,left=.05, right=.95)

            else:
                if not extra_space:
                    fig.subplots_adjust(wspace=0.05, hspace = 0.25,top=0.9,bottom=0.25,left=.05, right=.95)

                else:
                    fig.subplots_adjust(wspace=0.05, hspace = 0.25,top=0.9,bottom=0.1,left=.05, right=.95)

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


            path = "plot_" + fix + ".png"

            fig.savefig(path)
            plt.show()
            plt.close("all")