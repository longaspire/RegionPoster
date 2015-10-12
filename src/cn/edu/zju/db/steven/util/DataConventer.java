package cn.edu.zju.db.steven.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class DataConventer {
	
	public static void Conventor(String filename, String writeFile){
		File file = new File(filename);
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(file));
            String tempString = null;
            StringBuffer sb= new StringBuffer("");
            while ((tempString = reader.readLine()) != null) {
                String fileLine = "";
                //System.out.println(tempString);
                String[] items = tempString.split("\t");
                //System.out.println(items[0]);
                //System.out.println(items[1]);
                //System.out.println(items[2]);
                String vectorString = convent2Vector(items[1]);
                //System.out.println(vectorString);
                fileLine = items[2] + "  " + vectorString;
                System.out.println(fileLine);
                sb.append(fileLine + "\n");
            }
            FileWriter writer = new FileWriter(writeFile);
            BufferedWriter bw = new BufferedWriter(writer);
            bw.write(sb.toString());
           
            bw.close();
            writer.close();
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                }
            }
        }
	}

	private static String convent2Vector(String string) {
		// TODO Auto-generated method stub
		System.out.println(string);
		String[] items = string.split(" ");
		String temp = "";
		int i = 0;
		for(i = 0; i < (items.length - 1); i++){
			temp += (i+1) + ":" + items[i] + " ";
		}
		temp += (i+1) + ":" + items[i];
		return temp;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
	}

}
