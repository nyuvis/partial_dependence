
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
import scipy.interpolate as si
import sys

def plot_pdp(dfTest,
             model,
             featureIndex,
             numSamples,
             clustNumber,
             scale,
             shift,
             labels_name,
             thresh = 0.5,
             LBKeogh=True,
             spaghetti = True, 
             ChunksYes = False):

    def pdp(fix, rows, chosen_row):
        t = time.time()
        numRows = len(rows)
        newMatrixF = np.zeros((numRows,numSamples,numFeat))
        sampleVals = dfSample[fix]
        depthIndex = 0
        i = 0
        for r in rows:
            if float(i+1)%10000==0:
                print ("---- loading matrix: ", np.round(i/float(numRows),decimals=2)*100,"%")
                print ("------ elapsed: ",int(int(time.time()-t)/60), "m")

            i+=1
            indexHeight = 0
            for v in sampleVals:
                new_r = np.copy(r)
                new_r[dictLabtoIndex[fix]] = v
                newMatrixF[depthIndex][indexHeight] = new_r
                indexHeight+=1
            depthIndex+=1
        chosen_rowS = []
        for v in sampleVals:
            arow = np.copy(chosen_row)
            arow[dictLabtoIndex[fix]] = v
            chosen_rowS.append(arow)
        return newMatrixF,np.array(chosen_rowS)

    def computePred(matrixChangedRows,chosen_rowS):
        t = time.time()
        numRows= len(matrixChangedRows)
        pred_matrix = np.zeros((numRows,numSamples))
        matrixChangedRows = matrixChangedRows.reshape((numRows*numSamples, numFeat))
        ps = model.predict_proba(matrixChangedRows)
        ps = [x[data_set_pred_index] for x in ps]
        k = 0
        for i in range(0,numRows*numSamples):
            if i%numSamples ==0:
                pred_matrix[k] = ps[i:i+numSamples]
                k+=1
        chosen_rowS_Pred = model.predict_proba(chosen_rowS)
        chosen_rowS_Pred = [x[data_set_pred_index] for x in chosen_rowS_Pred]
        return pred_matrix, chosen_rowS_Pred

    def computePredChunks(matrixChangedRows,chosen_rowS):
        t = time.time()
        numRows= len(matrixChangedRows)
        pred_matrix = np.zeros((numRows,numSamples))
        for i in range(0,numRows):
            if float(i+1)%1000==0:
                print ("---- loading preds: ", np.round(i/float(numRows),decimals=4)*100,"%")
                print ("------ elapsed: ",int(int(time.time()-t)/60), "m")

            ps = model.predict_proba(matrixChangedRows[i])
            ps = [x[data_set_pred_index] for x in ps]
            pred_matrix[i] = ps
        chosen_rowS_Pred = model.predict_proba(chosen_rowS)
        chosen_rowS_Pred = [x[data_set_pred_index] for x in chosen_rowS_Pred]
        return pred_matrix, chosen_rowS_Pred


    def rmse(curve1, curve2):
        return np.sqrt(((curve1 - curve2) ** 2).mean())

    def LB_Keogh(s1,s2,r):
        LB_sum=0
        for ind,i in enumerate(s1):

            lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

            if i>upper_bound:
                LB_sum=LB_sum+(i-upper_bound)**2
            elif i<lower_bound:
                LB_sum=LB_sum+(i-lower_bound)**2

        return np.sqrt(LB_sum)

    def plottingPredictionChanges(pred_matrix,fix,labels_clust,clustNumber,rWarped,allClust,spag = False, pred_spag = None):

        featLol = fix
        original_data_sample = back_to_the_orginal(list(dfSample[fix]),fix)
        path = "plot_"+featLol
        #path = "plot_"+str(int(time.time()))[-3:]+"_"
        if rWarped>0:
            path = path+"_warped_"+str(rWarped)
        if allClust:
            path = path+"_all"
        path = path + ".png"
        t = time.time()
        numRows = len(pred_matrix)
        fig,ax = plt.subplots(figsize=(16, 9), dpi=300)
        plt.title("1D partial dependency of "+featLol,fontsize=20)
        
        clusters = np.array(range(clustNumber))/float(clustNumber)
        
        texts1 = []
        texts2 = []
        sizeClusts =[]
        all_indexes_in_cluster = {}
        for i in range(clustNumber):
            all_indexes_in_cluster[i] = [idI for idI in range(numRows) if labels_clust[idI] == i]
            
            #colors_labs.append(cmap(clusters[i]))
            is_index = []
            js_index = []
            for comb in combinations(all_indexes_in_cluster[i], 2):
                is_index.append(comb[0])
                js_index.append(comb[1])
            if len(all_indexes_in_cluster[i]) == 1:
                avgRmse = 0
            else:
                avgRmse = np.round(np.mean(distance_matrix[is_index,js_index]),decimals=3)
            sizeClust = len(all_indexes_in_cluster[i])
            sizeClusts.append(sizeClust)
            texts1.append("#"+str(i)+" - avg dist: "+str(avgRmse))
            texts2.append("#"+str(i)+" - size: "+str(sizeClust))
        
        
        colors_10_cluster = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
        colors_10_cluster = colors_10_cluster[1::2]+colors_10_cluster[::2]

       

        the_index_to_use = [x for _,x in sorted(zip(sizeClusts,range(clustNumber)),reverse=True)]

        colors_10_cluster = [x for _,x in sorted(zip(the_index_to_use,colors_10_cluster),reverse=False)]


        patches1 = [ plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors_10_cluster[i], 
                label="{:s}".format(texts1[i]) )[0]  for i in the_index_to_use ]
        patches2 = [ plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors_10_cluster[i], 
                label="{:s}".format(texts2[i]) )[0]  for i in the_index_to_use ] 
        
        legend1 = plt.legend(handles=patches1, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1, facecolor="#d3d3d3", numpoints=1,fontsize=13 )
        legend2 = plt.legend(handles=patches2, bbox_to_anchor=(1.04,0), loc="lower left", ncol=1, facecolor="#d3d3d3", numpoints=1,fontsize=13 )
        
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        
        #for i in range(numRows):
            #if float(i+1)%1000==0:
                #print ("---- loading plot: ", np.round(i/float(numRows),decimals=2)*100,"%")
                #print ("------ elapsed: ",int(int(time.time()-t)/60), "m")

            
            #plt.plot(dfSample[fix],pred_matrix[i],color=colors_10_cluster[labels_clust[i]],alpha=trasparenza)
            #plt.plot(dfSample[fix],pred_matrix[i],color=cmap(clusters[labels_clust[i]]),alpha=trasparenza)
            
        mean_preds = np.array([np.mean(pred_matrix[:,i]) for i in range(numSamples)])
        std_preds = np.array([np.std(pred_matrix[:,i]) for i in range(numSamples)])
        
        for i in range(clustNumber):
            if not spag:  
                x_point = x_array[all_indexes_in_cluster[i],dictLabtoIndex[fix]]
                x_point = back_to_the_orginal(x_point,fix)
                y_point = original_preds[all_indexes_in_cluster[i]] 
                plt.scatter(x_point,y_point,c=colors_10_cluster[i],s=dot_size)

            mean_predsNow = np.array([np.mean(pred_matrix[all_indexes_in_cluster[i],idf]) for idf in range(numSamples)])
            std_predsNow = np.array([np.std(pred_matrix[all_indexes_in_cluster[i],idf]) for idf in range(numSamples)])
            #ax.plot(dfSample[fix],mean_predsNow,color=colors_10_cluster[i],alpha=1)
            ax.fill_between(original_data_sample, mean_predsNow-std_predsNow, mean_predsNow+std_predsNow,color=colors_10_cluster[i],alpha=0.25)
            if spag:
                #print("plotting spaghetti cluster#",i)
                for j in all_indexes_in_cluster[i]:
                    #print("plotting spaghetto #",j)
                    NowSample =allSamplesOriginal[fix+"-o-"+str(j)][originalIndex-howFarIndex:originalIndex+howFarIndex+1]
                    NowPreds = pred_spag[j,originalIndex-howFarIndex:originalIndex+howFarIndex+1]
                    #plt.plot(NowSample,NowPreds,alpha=trasparenza,color=colors_10_cluster[i])
                    x_i,y_i = B_spline(np.array(NowSample),np.array(NowPreds))
                    plt.plot(x_i,y_i,alpha=0.8,color=colors_10_cluster[i])


        
        #plt.plot(dfSample[fix],mean_preds,color="red",alpha=1)
        #plt.fill_between(dfSample[fix], mean_preds-std_preds, mean_preds+std_preds,color="green",alpha=0.25)
        #plt.plot(dfSample[fix],chosen_rowS_Pred,color="red",alpha=1)
        plt.axvline(x=back_to_the_orginal(dfFeatures["mean"][fix],fix),color="green",linestyle='--')
        plt.axhline(y=thresh,color="red",linestyle='--')
        plt.ylabel("prediction",fontsize=20)

        plt.xlabel(featLol,fontsize=20)
        plt.xlim([original_data_sample[0],original_data_sample[numSamples-1]])
        plt.ylim([0,1])
        #pred = 0
        ax.text(-0.09,0.05,labels_name[1-data_set_pred_index],fontsize=20,transform=ax.transAxes)
        #pred = 1
        ax.text(-0.09,0.95,labels_name[data_set_pred_index],fontsize=20,transform=ax.transAxes)

        # no sides
        #plt.tight_layout()
        
        # only right side
        #plt.subplots_adjust(top=.9, bottom=.1, left=.05, right=.80)
        
        # both sides
        plt.subplots_adjust(top=.9, bottom=.1, left=.075, right=.80)
        
        # only left side
        #plt.subplots_adjust(top=.9, bottom=.1, left=.085, right=.99)
        # does not work: ax.margins(y=0.05)
        fig.savefig(path)
        plt.show()

        plt.close("all")
        
    def B_spline(x,y):

        n_local_points = len(x)
        t = range(n_local_points)
        ipl_t = np.linspace(0.0, n_local_points - 1, 100)

        x_tup = si.splrep(t, x, k=3)
        y_tup = si.splrep(t, y, k=3)

        x_list = list(x_tup)
        xl = x.tolist()
        size_seros = len(x_list[1])-len(xl)
        x_list[1] = xl + np.zeros(size_seros).tolist()

        y_list = list(y_tup)
        yl = y.tolist()
        size_seros = len(y_list[1])-len(yl)
        y_list[1] = yl + np.zeros(size_seros).tolist()

        x_i = si.splev(ipl_t, x_list)
        y_i = si.splev(ipl_t, y_list)
        return x_i,y_i

    def pdp_local(fix, rows, chosen_row,allSamples):
        t = time.time()
        numRows = len(rows)
        print("changing up",numRows,"rows")
        newMatrixF = np.zeros((numRows,numSamples+1,numFeat))
        depthIndex = 0
        i = 0
        for r in rows:
            
            sampleVals = local_sampling(fix,r,numSamples)
            allSamples[fix+"-o-"+str(i)] = sampleVals
            if float(i+1)%10000==0:
                print ("---- loading matrix: ", np.round(i/float(numRows),decimals=2)*100,"%")
                print ("------ elapsed: ",int(int(time.time()-t)/60), "m")

            i+=1
            indexHeight = 0
            
            for v in sampleVals:
                new_r = np.copy(r)
                new_r[dictLabtoIndex[fix]] = v
                newMatrixF[depthIndex][indexHeight] = new_r
                indexHeight+=1
            depthIndex+=1
        chosen_rowS = []
        for v in sampleVals:
            arow = np.copy(chosen_row)
            arow[dictLabtoIndex[fix]] = v
            chosen_rowS.append(arow)
        return newMatrixF,np.array(chosen_rowS),allSamples

    def computePred_local(matrixChangedRows,chosen_rowS):
        numS = numSamples + 1
        t = time.time()
        numRows= len(matrixChangedRows)
        pred_matrix = np.zeros((numRows,numS))
        matrixChangedRows = matrixChangedRows.reshape((numRows*numS, numFeat))
        ps = model.predict_proba(matrixChangedRows)
        ps = [x[data_set_pred_index] for x in ps]
        k = 0
        for i in range(0,numRows*numS):
            if i%numS ==0:
                pred_matrix[k] = ps[i:i+numS]
                k+=1
        chosen_rowS_Pred = model.predict_proba(chosen_rowS)
        chosen_rowS_Pred = [x[data_set_pred_index] for x in chosen_rowS_Pred]
        return pred_matrix, chosen_rowS_Pred


    def local_sampling (fix, chosen_row, numSamples):
        base_value = chosen_row[dictLabtoIndex[fix]]
        samplesLeft = list(np.linspace(base_value-1,base_value,int(numSamples/2)+1))
        samplesRight = list(np.linspace(base_value,base_value+1,int(numSamples/2)+1))
        samples = samplesLeft+samplesRight
        divisor = int(numSamples/2+1)
        final_samples = samples[:divisor-1]+[base_value]+samples[divisor+1:]
        return final_samples
        
    def back_to_the_orginal(data_this,fix):
        integ = dictLabtoIndex[fix]
        data_that = data_this / scale[0][integ] - shift[0][integ]
        return data_that

    scale = np.array(scale)
    shift = np.array(shift)
    if spaghetti:
        originalIndex = int(numSamples/2)
        howFarIndex = 2
        
    ###### intro ######
    data_set_pred_index = 0
    lenTest = len(dfTest)
    numFeat = len(dfTest.columns)-1
    ylabel = [dfTest.columns[-1]]
    xlabel = list(dfTest.columns[:numFeat])
    x_array = dfTest[xlabel].as_matrix()
    y_array = dfTest[ylabel].as_matrix()
    labs = [True if l[0] in [labels_name[0]] else False for l in y_array]
    ###################

    x_array = (x_array + shift)*scale

    pred = model.predict_proba(x_array)
    original_preds = np.array([x[data_set_pred_index] for x in pred])
    #thresh = get_roc_curve(original_preds,labs)["score"]
    

    dictLabtoIndex = {}
    i =0
    for laballa in xlabel:
        dictLabtoIndex[laballa] = i
        i+=1
        
    dfFeatures = pd.DataFrame(columns=["max","min","mean","sd"],index=xlabel)

    means = []
    stds = []
    mins = []
    maxs = []
    for laballa in xlabel:
        THErightIndex = dictLabtoIndex[laballa]
        vectzi = (list(dfTest[laballa])+ shift[0][THErightIndex])*scale[0][THErightIndex]
        mean = np.mean(vectzi)
        maxim = max(vectzi)
        minion = min(vectzi)
        standin = np.std(vectzi)
        means.append(mean)
        stds.append(standin)
        mins.append(minion)
        maxs.append(maxim)
    dfFeatures["max"] = maxs
    dfFeatures["min"] = mins
    dfFeatures["mean"] = means 
    dfFeatures["sd"] = stds
    numFeat = len(xlabel)

    dfSample = pd.DataFrame(columns=xlabel)

    eps = 0.01
    for laballa in xlabel:
        lower_bound = dfFeatures["min"][laballa] - eps
        higher_bound = dfFeatures["max"][laballa] + eps
        #bound = dfFeatures["mean"][laballa] + 2*dfFeatures["sd"][laballa]
        dfSample[laballa] = np.linspace(lower_bound,higher_bound ,numSamples)





    print ("--> ",i,": ",xlabel[featureIndex])
    the_feature = xlabel[featureIndex]

    print ("partial dependency for: ", the_feature )

    if not LBKeogh:
        print("We will be using RMSE then.")
    else:

        distance_between_2_samples = (dfSample[the_feature][numSamples-1]-dfSample[the_feature][0])/numSamples
        sugg_r = int(np.ceil(dfFeatures["sd"][the_feature]/10.0/distance_between_2_samples))
        print("Suggested r parameter:",sugg_r)
        #rWarpedUsed = int(input("warp parameter window:"))
        rWarpedUsed = sugg_r
        if rWarpedUsed == 0:
            rWarpedUsed =1

    changing_rows = np.copy(x_array)

    chosen_row = np.array(dfFeatures["mean"])
    the_matrix,chosen_rowS = pdp(the_feature,changing_rows,chosen_row)

    if ChunksYes:
        preds,chosen_rowS_Pred = computePredChunks(the_matrix,chosen_rowS)
    else:
        preds,chosen_rowS_Pred = computePred(the_matrix,chosen_rowS)
        
    list_of_test_indexes = range(lenTest)
    pairs_of_curves = []
    for comb in combinations(list_of_test_indexes, 2):
        pairs_of_curves.append(comb)

    k = 0
    all_total = len(pairs_of_curves)
    distance_matrix = np.zeros((lenTest,lenTest))
    start_time = time.time()
    for pair in pairs_of_curves:
        i = pair[0]
        j = pair[1]
        if k > 100:
            elapsed = time.time() - start_time
            perc = np.round(k/float(all_total)*100, decimals=2)
            elapsed = time.time() - start_time
            togo = 100 - perc
            towait = np.around( (elapsed*togo/perc) / 60.0, decimals = 2)
            sys.stdout.write("\r{0}% - waiting time: {1:.3f}m".format(perc, towait))
        k+=1
        if LBKeogh:
            distance = LB_Keogh(preds[i],preds[j],rWarpedUsed)
        else:
            distance = rmse(preds[i],preds[j])

        distance_matrix[i,j] = distance
        distance_matrix[j,i] = distance
        distance_matrix[i,i] = 0.0
    print()
    print("elapsed: ",np.around( (time.time() - start_time) / 60.0, decimals = 2),"m")
    #clust = AgglomerativeClustering(affinity='precomputed', n_clusters=clustNumber, linkage='complete')
    clust = AgglomerativeClustering(affinity='precomputed', n_clusters=clustNumber, linkage='average')
    #clust = AgglomerativeClustering(affinity='euclidean', n_clusters=clustNumber, linkage='ward')
    clust.fit(distance_matrix)
    #clust.fit(preds) #just if affinity='euclidean' and linkage='ward'
    labels_clust = clust.labels_

    trasparenza = 1
    dot_size = 5

    #cmap = plt.get_cmap("gist_rainbow")
    #cmap = plt.get_cmap("RdYlBu")
    # http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=10

    colors_10_cluster = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c',
                         '#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']

    if not spaghetti:
        if LBKeogh:
            plottingPredictionChanges(preds,the_feature,labels_clust,clustNumber,rWarped=rWarpedUsed,allClust=False)
        else:
            plottingPredictionChanges(preds,the_feature,labels_clust,clustNumber,rWarped=0,allClust=False)
    else:

        allSamples = {}

        changing_rows_local = np.copy(x_array)

        chosen_row_local = np.array(dfFeatures["mean"])
        the_matrix_local,chosen_rowS_local,allSamples = pdp_local(the_feature,changing_rows_local,chosen_row_local,allSamples)

        preds_local,chosen_rowS_Pred_local = computePred_local(the_matrix_local,chosen_rowS_local)

        allSamplesOriginal = {}
        for key in allSamples:
            allSamplesOriginal[key] = back_to_the_orginal(allSamples[key],key.split("-o-")[0])

        if LBKeogh:
            plottingPredictionChanges(preds,the_feature,labels_clust,clustNumber,rWarped=rWarpedUsed,allClust=False, spag = True, pred_spag = preds_local)
        else:
            plottingPredictionChanges(preds,the_feature,labels_clust,clustNumber,rWarped=0,allClust=False,spag = True, pred_spag = preds_local)

