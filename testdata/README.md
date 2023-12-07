# pygenlib test data

Pygenlib tests use various test data files that can be created by running the testdata.py script.
This script contains a config dict that describes the various test resources and their origin.
Briefly, it does the following for each configured resource (see code for details):
* Download source file from a public URL or copy from the static_files directory
* Ensure that the files are sorted by genomic coordinates and are compressed and indexed with bgzip and tabix.
* Slice a genomic subregions from the files if configured
* Copy the result files and corresponding indices to the testdata directory

For accessing test resources (including pybedtools test resources), you can use the testdata.get_resource() method.
For listing all available test resources, you can use the testdata.list_resources() method.

