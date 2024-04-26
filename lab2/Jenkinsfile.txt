pipeline{
    agent any

    stages{
        stage('Setup Environment'){
            steps{ sh 'pip install -r requirements.txt'
             }
        }
        stage('Download Data'){
            steps {
                 sh 'python3 data_creation.py'
            }
        }
        stage( 'Preprocess Data'){
            steps{
                sh 'python3 model_prep.py'
            }
        }
        stage('Train Model'){
            steps{
                sh 'python3 model.py'
            }
        }
        stage('Test Model'){
            steps{
                sh 'python3 model_testing.py'
            }
        }
    }
    post {
        always{
            sh 'echo "Pipeline completed."'
        }
    }
}

