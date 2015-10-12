package cn.edu.zju.db.steven.svm;

import java.io.IOException;

import cn.edu.zju.db.steven.util.DataConventer;

public class LibSVMTester {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		String trainFile = "radiomap";
		String testFile = "test";
		String resfile = "test_res";
		DataConventer dConventer = new DataConventer();
		
		dConventer.Conventor("E:\\BaiduYun\\Kanbox\\9Projects\\Room Determination\\0dataset_fingerprints\\fingerprint_Tue_Dec_03_211205_CST_2013.txt", trainFile);
		dConventer.Conventor("E:\\BaiduYun\\Kanbox\\9Projects\\Room Determination\\0dataset_fingerprints\\fingerprint_Mon_Dec_02_085459_CST_2013.txt", testFile);
		
		//dConventer.Conventor("UCI-breast-cancer-tra");
		
		String[] trainArgs = {testFile}; //{"UCI-breast-cancer-tra"};//directory of training file  
        String modelFile = svm_train.main(trainArgs);  
        String[] testArgs = {trainFile, modelFile, resfile};//directory of test file, model file, result file
		Double accuracy = svm_predict.main(testArgs);
		System.out.println("SVM Classification is done! The accuracy is " + accuracy);
	}

}
