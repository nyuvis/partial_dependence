partial_dependence
==================

A python library for plotting partial dependence patterns of machine learning classifiers.
The technique is a black box approach to recognize sets of instances where the model makes similar decisions.

Partial dependence measures the prediction change when changing one or more input features.
We will focus only on 1D and 2D partial dependence plots. 
For each instance in the data we can plot the prediction change as we change one or two features in defined sample ranges.
Then we cluster similar plots or heatmaps, e.g., instances reacting similarly when a feature value changes, to reduce clutter.

You can install *partial_dependence* via

.. code:: bash

    pip install partial_dependence

and import it in python using:

.. code:: python

    import partial_dependence as pdp_plot



********************************************
1. Plotting clustering of partial dependence
********************************************

Following we will show how the pipeline of functions works. Please refer to the inline documentation of the methods for full information.

You can also run the Jupyter notebook file to have a running example. 

The visualization we are using as example are coming from a Random Forest model trained on the `UCI Wine Quality Data Set <https://archive.ics.uci.edu/ml/datasets/wine+quality>`_.
The prediction is towards the class "good wine".

1.1 Initialization
##################

Required arguments:
*******************

* ``df_test``: a ``pandas.DataFrame`` containing only the features 
  values for each instance in the test-set. 
* ``model``: trained classifier as an object with the following properties. 
  
  The object must have a method ``predict_proba(X)`` which takes a ``numpy.array`` of shape ``(n, num_feat)`` as input and returns a ``numpy.array`` of shape ``(n, len(class_array))``.

* ``class_array``: a list of strings with all the classes name in the same order 
  as the predictions returned by ``predict_proba(X)``.
* ``class_focus``: a string with the class name of the desired partial dependence.

Optional arguments:
*******************

* ``num_samples``: number of desired samples. Sampling a feature is done with:

  ``numpy.linspace(min_value, max_value, num_samples)``

  where the bounds are related to min and max value for that feature in the test-set. Default value is 100.
* ``scale``: scale parameter vector for normalization.
* ``shift``: shift parameter vector for normalization.

If you need to provide your data to the model in normalized form, 
you have to define scale and shift such that: 

``transformed_data = (original_data + shift)*scale``

where ``shift`` and ``scale`` are both ``numpy.array`` of shape ``(1,num_feat)``.

If the model uses directly the raw data in ``df_test`` without any transformation, 
do not insert any scale and shift parameters.

If our model does not use normalization, we can initialize the tool this way:


.. code:: python

    my_pdp_plot = pdp_plot.PartialDependence( my_df_test,
                                              my_model,
                                              my_labels_name,
                                              my_labels_focus )



1.2 Creating the PdpCurves object
#################################

By choosing a feature and changing it in the sample range, for each row in the test-set we can create ``num_samples`` different versions of the original instance.

Then we are able to compute prediction values for each of the different vectors.

``pdp()`` initialize and returns a python object from the class ``PdpCurves()`` containing such predictions values.


Required argument:
******************

* ``fix``: string with name of the chosen feature as reported in a column of ``df_test``.


.. code:: python

    curves = my_pdp_plot.pdp( chosen_feature )

1.3 Getting an overview of the partial dependence
#################################################

It is already possible to plot something with the function ``plot()``.

Whenever you have a ``PdpCurves`` object available, you can plot something.
Here you can find a first example. The visualization is automatically saved in a png file in the same folder of the script.

.. code:: python

    my_pdp_plot.plot( curves, local_curves = True, plot_full_curves = True )

.. image:: images/full_curves.png
    :width: 1600px
    :align: center
    :alt: alternate text

1.4 Clustering 1D partial dependence
####################################

To call ``compute_clusters()``, we define the integer number of desired clusters with the ``n_clusters`` argument and we provide ``curves``.

The function returns a list of ``PdpCurves`` objects. Each element of the list is a different cluster.

.. code:: python

    curves_list_RF = my_pdp_plot.compute_clusters( curves, chosen_cluster_number )


1.5 Plotting the clustering results
###################################

Without customization, plotting the clustering is quite straightforward.

.. code:: python

    my_pdp_plot.plot( curves_list_RF )

.. image:: images/clustering.png
    :width: 1600px
    :align: center
    :alt: alternate text

1.6 2D partial dependence heatmaps
##################################

It is possible to visualize the increase/decrease in prediction of instances when changing two features at the same time.
For a single instance the samples vary around the original pair of values.
You can specify the desired instance by providing the row index integer from ``df_test``.
In this case we are taking the instance with index 88.

.. code:: python

    instance_heatmap = my_pdp_plot.pdp_2D("alcohol", "density", instances = 88)
    my_pdp_plot.plot_heatmap(instance_heatmap)

.. image:: images/single.png
    :width: 1080px
    :align: center
    :alt: alternate text

In case you want to visualize the average 2D partial dependence over a set of instances, just provide a list of integers.
The color will resemble the average increase/decrease across all instances and the samples will vary from min to max values of the set.
If you want to visualize the average 2D partial dependence across the entire test-set instead..

.. code:: python

    all_inst = my_pdp_plot.pdp_2D("alcohol", "density")
    my_pdp_plot.plot_heatmap(all_inst)

.. image:: images/heatmap_test.png
    :width: 1080px
    :align: center
    :alt: alternate text

1.7 Clustering 2D partial dependence
####################################

With same function ``my_pdp_plot.compute_clusters()`` of Section 1.4, it is also possible to cluster heatmaps. 

An heatmap object from the command ``my_pdp_plot.pdp_2D(feat_y, feat_x, instances)`` contains: 
``num_samples`` X ``num_samples`` X ``len(instances)`` prediction values.

It is possible to cluster all the test instances (using the RMSE metric) and to display an heatmaps for each cluster with the following code:

.. code:: python

    all_inst = my_pdp_plot.pdp_2D("alcohol", "density")
    list_clust_heats = my_pdp_plot.compute_clusters(all_inst, n_clusters = 16)
    my_pdp_plot.plot_heatmap(list_clust_heats)

.. image:: images/clust_heats_test.png
    :width: 1080px
    :align: center
    :alt: alternate text

1.8 2D partial dependence SPLOMs
################################

We can combine all the possible heatmaps in a single visualization.
The SPLOM will show the patterns describing all possible pairs of features partial dependence.

The code to visualize the SPLOM for that same instance 88 is quite simple:

.. code:: python

    sploms_objs = my_pdp_plot.get_data_splom(88)
    my_pdp_plot.plot_splom(sploms_objs)

A stripe of blue/red over a column and row of a feature determines an increase/decrease of prediction when that feature is changed, no matter what other feature varies.
For example for this particular instance, when changing just two features, an increase in *alcohol* or decrease in *volatile acidity* would generally bring an increase in prediction towards the class *good wine*.

.. image:: images/single_splom.png
    :width: 1080px
    :align: center
    :alt: alternate text

The SPLOM can give you a hint of average prediction change also over the entire test-set.
The visualization combines the 2D scatter plots with the average change in prediction. 

The user can detect global patterns when a same color disposition is present across row and columns of a same feature.
For example this model generally has an average increase in prediction towards the class *good wine* when the *alcohol* increases with any other feature.
Dark orange areas and blue areas show where there is an average decrease/increase in prediction.
For example there is an enclaved blue area within the heatmap cell for *pH* and *total sulfur dioxide* where the prediction generally increases.

.. code:: python

    sploms_objs = my_pdp_plot.get_data_splom()
    my_pdp_plot.plot_splom(sploms_objs)

.. image:: images/splom_test.png
    :width: 1080px
    :align: center
    :alt: alternate text

1.9 Clustering SPLOMs
#####################

Each instance SPLOM can be represented by a long vector of prediction values.
The vector is created by appending the data from each unique heatmap in a SPLOM.
We can measure the distance among different instances SPLOMs by computing RMSE among such vectors.
By building an RMSE distance matrix and clustering the instances we are able to represent a SPLOM for each cluster set.
With the following code we can cluster the SPLOMs of the entire test-set. 

.. code:: python

    sploms_objs = my_pdp_plot.get_data_splom()
    list_clust_sploms = my_pdp_plot.compute_clusters(sploms_objs, n_clusters = 16)


To have an overview over the entire set of clusters:

.. code:: python

    my_pdp_plot.plot_splom(list_clust_sploms)

.. image:: images/cluster_sploms.png
    :width: 1080px
    :align: center
    :alt: alternate text

We can now plot the first cluster (cluster with label "#8" in the left top corner of the last viz)

.. code:: python

    my_pdp_plot.plot_splom(list_clust_sploms[0])


.. image:: images/first_cluster_splom.png
    :width: 1080px
    :align: center
    :alt: alternate text

The distance matrix is stored, so it is less time consuming to change the number of clusters and plot again.

.. code:: python

    list_clust_sploms = my_pdp_plot.compute_clusters(sploms_objs, n_clusters = 49)
    my_pdp_plot.plot_splom(list_clust_sploms)

.. image:: images/cluster_sploms_49.png
    :width: 1080px
    :align: center
    :alt: alternate text


****************************************
2. Customization and extra functions
****************************************

2.1 Computing predictions in chunks
###############################

When using ``pdp()``, sometimes the amount of data to process is too large and it is necessary to divide it in chunks so that we don't run out of memory.
To do so, just set the optional argument ``batch_size`` to the desired integer number. 

``batch_size`` cannot be lower than ``num_samples`` or higher than ``num_samples * len(df_test)``. 
If ``batch_size`` is 0, then the computation of prediction will take place in a single chunk, which is much faster if you have enough memory.

.. code:: python

    curves = my_pdp_plot.pdp( chosen_feature, batch_size = 1000 )


2.2 Using your own matplotlib figure
################################

If you really like to hand yourself matplotlib and be free to customize the visualization this is how it works:

.. code:: python

    curves_list_RF = my_pdp_plot.compute_clusters(curves, chosen_cluster_number)

    cluster_7 = curves_list_RF[7]
    cluster_0 = curves_list_RF[0]
    cluster_9 = curves_list_RF[9]

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

    my_pdp_plot.plot(cluster_7,
                       color_plot="red", 
                       plot_object=ax)

    my_pdp_plot.plot(cluster_0,
                       color_plot="blue", 
                       plot_object=ax)

    my_pdp_plot.plot(cluster_9,
                       color_plot="green", 
                       plot_object=ax)

    plt.show()
    plt.close("all")

.. image:: images/own_figure.png
    :width: 1600px
    :align: center
    :alt: alternate text


2.3 Comparing different models
##############################

There might be scenarios in which you want to compare clusters from different models.
For example let's compare the Random Forest model we had so far with a Support Vector Machine model.

.. code:: python

    wine_pdp_plot_SVM = pdp_plot.PartialDependence(df_test,
                                                    model_SVM,
                                                    labels_name,
                                                    labels_focus,
                                                    num_samples,
                                                    scale_SVM,
                                                    shift_SVM)

    curves = wine_pdp_plot_SVM.pdp(chosen_feature)
    curves_list_SVM = wine_pdp_plot_SVM.compute_clusters(curves, chosen_cluster_number)
    wine_pdp_plot_SVM.plot(curves_list_SVM)

.. image:: images/SVM.png
    :width: 1600px
    :align: center
    :alt: alternate text


2.4 Clustering with DTW distance
################################

To cluster together the partial dependence plots, we measure the distance among each pair.
By default this distance is measured with RMSE.
Another option for 1D partial dependence clustering is `LB Keogh <http://www.cs.ucr.edu/~eamonn/LB_Keogh.htm>`_  distance, an approximation of Dynamic Time Warping (DTW) distance.
By setting the ``curves.r_param`` parameter of the formula to a value different from ``None``, you are able to compute the clustering with the LB Keogh.
The method ``get_optimal_keogh_radius()`` gives you a quick way to automatically compute an optimal value for ``curves.r_param``.
To set the distance back to RMSE just set ``curves.set_keogh_radius(None)`` before recomputing the clustering.

The first time you compute the clustering, a distance matrix is computed. 
Especially when using DTW distance, this might get time consuming.
After the first time you call ``compute_clusters()`` on the ``curves`` object, 
the distance matrix will be stored in memory and the computation will be then much faster.
Anyway if you change the radius with ``curves.set_keogh_radius()``, you will need to recompute again the distance matrix.

.. code:: python

    curves.set_keogh_radius( my_pdp_plot.get_optimal_keogh_radius() )
    keogh_curves_list = my_pdp_plot.compute_clusters( curves, chosen_cluster_number )

2.5 An example of the visualization customizations
##############################################

.. code:: python

    my_pdp_plot.plot( keogh_curves_list, local_curves = False, plot_full_curves = True )

.. image:: images/custom.png
    :width: 1600px
    :align: center
    :alt: alternate text

.. code:: python

    curves_list_RF = my_pdp_plot.compute_clusters( curves_RF, 5 )

    my_pdp_plot.plot( curves_list_RF, cell_view = True )

.. image:: images/RF_five_cell_view.png
    :width: 1600px
    :align: center
    :alt: alternate text

.. code:: python

    curves_list_SVM = my_pdp_plot_SVM.compute_clusters( curves_SVM, 25 )

    my_pdp_plot_SVM.plot( curves_list_SVM, 
                            cell_view = True, 
                            plot_full_curves = True, 
                            local_curves = False, 
                            path="plot_alcohol.png" )

.. image:: images/SVM_25_all.png
    :width: 1600px
    :align: center
    :alt: alternate text

2.6 Highlighting a custom vector
################################

In case you want to highlight the partial dependence of a particular vector ``custom_vect``, this is how it works..

.. code:: python

    curves, custom_preds = my_pdp_plot.pdp( chosen_feature, chosen_row = custom_vect )

    my_pdp_plot.compute_clusters( curves, chosen_cluster_number )

    my_pdp_plot.plot( curves, local_curves = False,
                       chosen_row_preds_to_plot = custom_preds )

.. image:: images/custom_vect.png
    :width: 1600px
    :align: center
    :alt: alternate text
