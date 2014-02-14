#  Copyright Matthias Troyer, Synge Todo and Lukas Gamper 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

file(WRITE tmp_${cmd}.sh "PYTHONPATH=\$PYTHONPATH:${pythonpath} ${python_interpreter} ${cmddir}/${cmd}")

find_file(input_path ${input}.input ${binarydir} ${sourcedir})
find_file(output_path ${output}.output ${binarydir} ${sourcedir})

if(input_path)
    execute_process(
        COMMAND sh tmp_${cmd}.sh
        RESULT_VARIABLE not_successful
        INPUT_FILE ${input_path}
        OUTPUT_FILE ${cmd}_output
        ERROR_VARIABLE err
        TIMEOUT 600
    )
else(input_path)
    execute_process(
        COMMAND sh tmp_${cmd}.sh
        RESULT_VARIABLE not_successful
        OUTPUT_FILE ${cmd}_output
        ERROR_VARIABLE err
        TIMEOUT 600
    )
endif(input_path)

file(REMOVE tmp_${cmd}.sh)

if(not_successful)
    message(SEND_ERROR "error runing test 'python_${cmd}': ${err}; shell output: ${not_successful}!")
endif(not_successful)

if(output_path)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E compare_files ${output_path} ${cmd}_output
        RESULT_VARIABLE not_successful
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        TIMEOUT 600
    )
    if(not_successful)
        message(SEND_ERROR "output does not match for 'python_${cmd}': ${err}; ${out}; shell output: ${not_successful}!")
    endif(not_successful)
endif(output_path)

file(REMOVE ${cmd}_output)
