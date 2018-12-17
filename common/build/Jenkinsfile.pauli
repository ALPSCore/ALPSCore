
/// Calling shell for the particular phase
def call_phase(phase, compiler, mpilib) {
    sh """export PHASE=${phase}
          export COMPILER=${compiler}
          export MPI_VERSION=${mpilib}
          ./common/build/build.pauli.jenkins.sh
       """
}

/// Sub-pipeline for a project; returns closure defining the sub-pipe
def sub_pipe(name, compiler, mpilib) {
    { ->
        stage("My stage with ${name}") {
            stash(name: name)
            node("master-node") {
                unstash(name: name)
                
                stage("Config") {
                    echo "Config step with compiler=${compiler} mpilib=${mpilib}"
                    call_phase('cmake', compiler, mpilib)
                }

                stage("Build") {
                    echo "Build step with compiler=${compiler} mpilib=${mpilib}"
                    call_phase('make', compiler, mpilib)
                }

                stage("Test")  {
                    echo "Test step with compiler=${compiler} mpilib=${mpilib}"
                    call_phase('test', compiler, mpilib)
                }
            }
        }
    }
}


pipeline {
    agent {
        node {
            label 'master-node'
        }

    }

    parameters {
        string(name: 'COMPILERS', defaultValue: 'gcc_4.8.5,gcc_5.4.0,clang_3.4.2,clang_5.0.1,intel_18.0.5', description: 'Compilers to use')
        string(name: 'MPI_VERSIONS', defaultValue: 'MPI_OFF,OpenMPI', description: 'MPI versions to link with')
    }

    stages {
        stage('Parallel stages') {
            steps {
                script {

                    projects = [:]
                    for (comp in params.COMPILERS.tokenize(',')) {
                        for (mpilib in params.MPI_VERSIONS.tokenize(',')) {

                            // Filter out combinations that don't work with MPI
                            if (comp=="gcc_5.4.0" || comp=="intel_18.0.5" || mpilib=="MPI_OFF") {
                                key="compiler=${comp}_mpilib=${mpilib}"
                                projects[key]=sub_pipe(key, comp, mpilib)
                            }

                        }
                    }
                    echo "DEBUG: Projects: ${projects}"
                    parallel (projects)

                } // end script
            } // end steps
        } // end stage
    } // end stages
}
