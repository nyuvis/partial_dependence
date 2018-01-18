partial_dependence
==================

A library for plotting partial dependency patterns of machine learning classfiers.
Partial dependence measures the prediction change when changing one or more input features.
We will focus only on 1D partial dependency plots. 
For each instance in the data we can plot the prediction change as we change a single feature in a defined sample range.
Then we cluster similar plots, e.g., instances reacting similarly value changes, to reduce clutter.
The technique is a black box approach to recognize sets of instances where the model makes similar decisions.

You can install *partial_dependence* via

.. code:: bash

    pip install partial_dependence

and import it in python using:

.. code:: python

	import partial_dependence as pdp_plot



****************************************
Plotting clustering of partial dependence
****************************************

Following we will show the pipeline of functions works. Please refer to the inline documentation of the methods for full information.

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

* ``num_samples``: number of desired samples. Sampling a feature is done with:
  ``numpy.linspace(min_value,max_value,num_samples)``, 
  where the bounds are related to min and max value for that feature in the test-set.
* ``scale``: scale parameter vector for normalization.
* ``shift``: shift parameter vector for normalization.

Instead if you need to provide your data to the model in normalized form, 
you have to define scale and shift such that: 
``transformed_data = (original_data + shift)*scale``,
where ``shift`` and ``scale`` are both ``numpy.array`` of shape ``(1,num_feat)``.
If the model uses directly the raw data in ``df_test`` without any transformation, 
do not insert any scale and shift parameters. 


.. code:: python

	my_pdp_plot = pdp_plot.PartialDependence( my_df_test,
							                  my_model,
							                  my_labels_name,
							                  my_labels_focus,
							                  my_number_of_samples,
							                  my_scale,
							                  my_shift )




