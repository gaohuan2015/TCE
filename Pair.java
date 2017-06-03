package model;

import java.io.Serializable;

public class Pair implements Serializable{
	int headId;
	int tailId;
	
	
	public Pair(int headId, int tailId) {
		super();
		this.headId = headId;
		this.tailId = tailId;
	}


	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + headId;
		result = prime * result + tailId;
		return result;
	}


	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Pair other = (Pair) obj;
		if (headId != other.headId)
			return false;
		if (tailId != other.tailId)
			return false;
		return true;
	}


	@Override
	public String toString() {
		return headId + " " + tailId;
	}


	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}


	

}
