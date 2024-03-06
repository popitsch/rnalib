#!/bin/bash
. /Users/niko/.virtualenvs/rnalib/bin/activate
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <python_script>"
    exit 1
fi
name=$1
if [ -f "${name}.py" ]; then
  terminalizer record -c config.yml --skip-sharing --command "python3 -c \"import rnalib as rna; rna.execute_screencast('${name}.py')\"" ${name}
  terminalizer render -o ${name}.gif ${name}.yml
  python3.10 -c "import pygifsicle; pygifsicle.optimize('${name}.gif')"
else
    echo "File ${name}.py does not exist."
    exit 1
fi
