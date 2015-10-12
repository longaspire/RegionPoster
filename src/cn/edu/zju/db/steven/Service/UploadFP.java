package cn.edu.zju.db.steven.Service;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.Calendar;
import java.util.List;

import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.tomcat.util.http.fileupload.FileItem;
import org.apache.tomcat.util.http.fileupload.FileItemFactory;
import org.apache.tomcat.util.http.fileupload.disk.DiskFileItemFactory;
import org.apache.tomcat.util.http.fileupload.servlet.ServletFileUpload;

/**
 * Servlet implementation class UploadFP
 */
@WebServlet(description = "Upload Fingerprints", urlPatterns = { "/UploadFP" })
public class UploadFP extends HttpServlet {
	private static final long serialVersionUID = 1L;

	/**
	 * @see HttpServlet#HttpServlet()
	 */
	public UploadFP() {
		super();
		// TODO Auto-generated constructor stub
	}

	/**
	 * @see HttpServlet#service(HttpServletRequest request, HttpServletResponse
	 *      response)
	 */
	protected void service(HttpServletRequest request,
			HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
		try {
			request.setCharacterEncoding("UTF-8"); // ���ô�����������ı����ʽ
			response.setContentType("text/html;charset=UTF-8"); // ����Content-Type�ֶ�ֵ
			PrintWriter out = response.getWriter();

			// ����Ĵ��뿪ʼʹ��Commons-UploadFile��������ϴ����ļ�����
			FileItemFactory factory = new DiskFileItemFactory(); // ����FileItemFactory����
			ServletFileUpload upload = new ServletFileUpload(factory);
			// �������󣬲��õ��ϴ��ļ���FileItem����
			List<FileItem> items = upload.parseRequest(request);
			// ��web.xml�ļ��еĲ����еõ��ϴ��ļ���·��
			System.out.println(request.getSession().getServletContext().getRealPath(""));
			String uploadPath = "E:\\BaiduYun\\Kanbox\\9Projects\\Room Determination\\0dataset_fingerprints\\";
			File file = new File(uploadPath);
			if (!file.exists()) {
				file.mkdir();
			}
			String filename = ""; // �ϴ��ļ����浽���������ļ���
			InputStream is = null; // ��ǰ�ϴ��ļ���InputStream����
			// ѭ�������ϴ��ļ�
			for (FileItem item : items) {
				// ������ͨ�ı���
				if (item.isFormField()) {
					if (item.getFieldName().equals("filename")) {
						// ������ļ���Ϊ�գ����䱣����filename��
						if (!item.getString().equals(""))
							filename = item.getString("UTF-8");
					}
				}
				// �����ϴ��ļ�
				else if (item.getName() != null && !item.getName().equals("")) {
					// �ӿͻ��˷��͹������ϴ��ļ�·���н�ȡ�ļ���
					filename = item.getName().substring(
							item.getName().lastIndexOf("\\") + 1);
					is = item.getInputStream(); // �õ��ϴ��ļ���InputStream����
				}
			}
			// ��·�����ϴ��ļ�����ϳ������ķ����·��
			String datetimeString = Calendar.getInstance().getTime().toString()
					.replaceAll(" ", "_").replaceAll(":", "");
			filename = uploadPath
					+ filename.substring(0, filename.length() - 4) + "_"
					+ datetimeString + ".txt";
			System.out.println(filename);
			// ����������Ѿ����ں��ϴ��ļ�ͬ�����ļ����������ʾ��Ϣ
			if (new File(filename).exists()) {
				new File(filename).delete();
			}
			// ��ʼ�ϴ��ļ�
			if (!filename.equals("")) {
				// ��FileOutputStream�򿪷���˵��ϴ��ļ�
				FileOutputStream fos = new FileOutputStream(filename);
				byte[] buffer = new byte[8192]; // ÿ�ζ�8K�ֽ�
				int count = 0;
				// ��ʼ��ȡ�ϴ��ļ����ֽڣ����������������˵��ϴ��ļ��������
				while ((count = is.read(buffer)) > 0) {
					fos.write(buffer, 0, count); // �������ļ�д���ֽ���

				}
				fos.close(); // �ر�FileOutputStream����
				is.close(); // InputStream����
				System.out.println("Success!");
				out.println("Uploading Successful");
				RequestDispatcher dispatcher = request.getRequestDispatcher("/LoadRadioMap?name=" + filename); 
				dispatcher.forward(request, response); 
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse
	 *      response)
	 */
	protected void doGet(HttpServletRequest request,
			HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse
	 *      response)
	 */
	protected void doPost(HttpServletRequest request,
			HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
		
	}

}
