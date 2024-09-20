.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/JarronL/pynrc/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

pyNRC could always use more documentation, whether as part of the
official pyNRC docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/JarronL/pynrc/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

..
        Get Started!
        ------------

        Ready to contribute? Here's how to set up `pynrc` for local development.

        1. Fork the `pynrc` repo on GitHub.
        2. Clone your fork locally::

            $ git clone git@github.com:your_name_here/pynrc.git

        3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper 
           installed, this is how you set up your fork for local development::

            $ mkvirtualenv pynrc
            $ cd pynrc/
            $ python setup.py develop

        4. Create a branch for local development::

            $ git checkout -b name-of-your-bugfix-or-feature

           Now you can make your changes locally.

        5. When you're done making changes, check that your changes pass flake8 and the
           tests, including testing other Python versions with tox::

            $ flake8 pynrc tests
            $ python setup.py test or py.test
            $ tox

           To get flake8 and tox, just pip install them into your virtualenv.

        6. Commit your changes and push your branch to GitHub::

            $ git add .
            $ git commit -m "Your detailed description of your changes."
            $ git push origin name-of-your-bugfix-or-feature

        7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring.
3. The pull request should work for Python 3.7, 3.8, and 3.9 for PyPy. Check https://travis-ci.org/JarronL/pynrc/pull_requests and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests:

.. code-block:: sh

   $ py.test tests.test_pynrc


Deploying
---------

A reminder for the maintainers on how to deploy. First, make sure the following packages are installed:

.. code-block:: sh

   $ pip install sphinx_automodapi
   $ pip install pandoc
   $ conda install sphinx_rtd_theme
   $ conda install nbsphinx
   $ conda install twine
   $ conda install docutils=0.18
   $ conda install bump2version



1. Add entries to HISTORY.rst. Make sure all your changes are committed to git.
2. Update version using ``bump2version``, which automatically updates ``pynrc.version``. Usage: ``bump2version [options] part [file]``, where "part" is either major, minor, or patch (e.g., major.minor.patch). See https://github.com/c4urself/bump2version for more details. For instance, ``bump2version minor`` will update from 1.0.5 to 1.1.0

   .. code-block:: sh

      $ bump2version [major | minor | patch]

3. Generate documentation locally:

   .. code-block:: sh

      $ make docs

4. Push all updates to github and make sure readthedocs generates correctly before actually submitting the release.
5. Package a distribution and test upload the release to TestPyPI:
   
   .. code-block:: sh

      $ make release-test

6. If everything works without a hitch, then upload the release to PyPI:

   .. code-block:: sh

      $ make release
    
   This command also tags the release on github. Make sure to have the command line token handy to enter as the requested password. Double-check ``stable`` release of readthedocs.
   
.. todo::

      7. Release code to ``conda-forge``. If you already have a conda-forge  feedstock forked to your own GitHub account, first edit ``recipe/meta.yaml`` to update the version, hash, etc. To calculate the sha256 hash, run:
   
         .. code-block:: sh

            openssl dgst -sha256 path/to/package_name-0.1.1.tar.gz
   
         Then, commit and push the yaml file to GitHub:
   
         .. code-block:: sh

            git pull upstream master
            git add --all
            git commit -m 'version bump to v0.1.1'
            git push -u origin master
   
         Finally, issue a pull request to conda-forge.
       
8. At end of all this, double-check the build environments at https://readthedocs.org/projects/pynrc/builds/. For whatever reason, it is common for there to be an OSError and the build to fail. Resetting the environment at https://readthedocs.org/projects/pynrc/versions/ tends to fix this issue. Build times take about 5 minutes.
       
.. Travis will then deploy to PyPI if tests pass.
