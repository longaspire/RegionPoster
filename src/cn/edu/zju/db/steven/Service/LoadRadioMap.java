package cn.edu.zju.db.steven.Service;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import cn.edu.zju.db.steven.svm.svm_train;
import cn.edu.zju.db.steven.util.DataConventer;

/**
 * Servlet implementation class LoadRadioMap
 */
@WebServlet(description = "load radio map", urlPatterns = { "/LoadRadioMap" })
public class LoadRadioMap extends HttpServlet {
	private static final long serialVersionUID = 1L;
       
    /**
     * @see HttpServlet#HttpServlet()
     */
    public LoadRadioMap() {
        super();
        // TODO Auto-generated constructor stub
    }

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
		String StringFilename = (String)request.getParameter("name");
		System.out.println("Test!" + StringFilename);
		
		DataConventer dConventer = new DataConventer();
		String trainFile = "radiomap";
		dConventer.Conventor(StringFilename, trainFile);
		//dConventer.Conventor("UCI-breast-cancer-tra");
		
		String[] trainArgs = {trainFile}; //{"UCI-breast-cancer-tra"};//directory of training file
		svm_train svmtrain = new svm_train();
        String modelFile = svm_train.main(trainArgs);  
        System.out.println(modelFile);
	}

	/**
	 * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
		doGet(request, response);
	}

}
