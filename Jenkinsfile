pipeline {
  agent {
    node {
      label 'master-node'
    }

  }
  stages {
    stage('Configure') {
      steps {
        sh '''export COMPILER=gcc_5.4.0
export MPI_VERSION=MPI_OFF
PHASE=cmake ./common/build/build.pauli.jenkins.sh
'''
      }
    }
  }
}