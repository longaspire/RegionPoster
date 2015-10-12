package cn.edu.zju.db.steven.Service;

import java.io.IOException;
import java.io.PrintWriter;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import cn.edu.zju.db.steven.svm.*;

/**
 * Servlet implementation class LabelMaker
 */
@WebServlet(description = "determine the room label", urlPatterns = { "/LabelMaker" })
public class LabelMaker extends HttpServlet {
	private static final long serialVersionUID = 1L;
       
    /**
     * @see HttpServlet#HttpServlet()
     */
    public LabelMaker() {
        super();
        // TODO Auto-generated constructor stub
    }

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
		doPost(request, response);
	}

	/**
	 * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
		String testData = (String)request.getParameter("data");
		System.out.println(testData);
		String res = svm_predict.predict(testData, "radiomap.model");
        
        System.out.println("the result is :" + res); 
        if(res != "-1")
        {
			response.setCharacterEncoding("gbk");
			PrintWriter out = response.getWriter();
			
			out.print("<?xml version=\"1.0\" encoding=\"gbk\"?>");
			out.print("<rp>");
			out.print("<code>");
			out.print("001");
			out.print("</code>");
			out.print("<res>");
			out.print("<svm>");
			out.print(res);
			out.print("</svm>");
			out.print("</res>");
			out.print("</rp>");
			out.flush();
			out.close();
		}else {
			response.setCharacterEncoding("gbk");
			PrintWriter out = response.getWriter();
			
			out.print("<?xml version=\"1.0\" encoding=\"gbk\"?>");
			out.print("<rp>");
			out.print("<code>");
			out.print("401");
			out.print("</code>");
			out.print("</rp>");
			out.flush();
			out.close();
		}
	}

}
