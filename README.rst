partial_dependence
==================

A library for plotting partial dependency patterns of machine learning classfiers.
Partial dependence measures a prediction change when changing one or more input features.
We will focus only on 1D partial dependency plots. 
For each istance in the test-set we can plot the prediction change as we change a single feature in a defined sample range.
Then we cluster together similar plots, e.g. istances which react similarly as we change that feature value.
The technique should give a black box approach to recognize set of istances where the model makes similar decisions.

You can install *partial_dependence* via

.. code:: bash

    pip install partial_dependence

and import it in python using:

.. code:: python

	import partial_dependence as pdp_plot



****************************************
Plotting clustering of partial dependence
****************************************

Following we will show the pipeline of function works. Please refer to the inline documentation of the methods for full information.

Initialization
##############

Required mandatory arguments:
****************************

* ``df_test``: a ``pandas.DataFrame`` containing only the features 
  values for each istance in the test-set. 
* ``model``: trained classfier as an object with the following properties. 
  The object must have a method ``prodict_proba(X)`` which takes a ``numpy.array`` of shape ``(n, num_feat)`` as input and returns a ``numpy.array`` of shape ``(n, len(class_array))``.
* ``class_array``: a list of strings with all the classes name in the same order 
  as the predictions returned by ``prodict_proba(X)``.
* ``class_focus``: a string with the class name of the desired partial dependence.

Optional arguments:
*******************

	* ``num_samples``: number of desired samples.
	* ``scale``: scale parameter for normalization.
	* ``shift``: shift parameter for normalization.

.. code:: python

	my_pdp_plot = pdp_plot.PartialDependence(my_df_test,
	                  my_model,
	                  my_labels_name,
	                  my_labels_focus,
	                  my_number_of_samples,
	                  my_scale,
	                  my_shift)


