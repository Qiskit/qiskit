===========================
Creating External Providers
===========================

Terra provides a plugin interface to define external backend providers


Creating a Plugin
-----------------

Creating a plugin is fairly straightforward and doesn't require much additional
effort on top of creating a provider. All external provider classes must be
defined off the abstract class defined at ``qiskit.backends.baseprovider``.

Entry Point
-----------

Once you've created your plugin class you need to add a `setuptools`_ entry
point to your provider to enable Qiskit-Terra to find the plugin. The entry
point must be added to the ``qiskit.providers`` namespace.

.. _setuptools: https://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins

For example, in your provider package's setup.py you would add something like::

   entry_points={'qiskit.providers':
      ['MyProvider = my_provider.provider:MyProvider']
   }

where ``my_provider.provider`` is the import path and ``MyProvider`` is the
provider class. Once this is added to your setup.py then whenever the package is
installed Qiskit-terra will detect the provider is installed and a
``MyProvider`` provider instance to the qiskit namespace.


Using Provider Plugins
----------------------

When installing provider plugins the name of the entry point becomes the name
of providers off the global instance of ``qiskit.providers`` namespace. So
continuing from the above example after installing the ``my_provider`` package
you can access the backends provided by ``MyProvider`` by using
``qiskit.providers.MyProvider``. Any other provider plugins installed will also
be accessible off of ``qiskit.providers``.
