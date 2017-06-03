package model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RelationPath {
	List<Integer> path = new ArrayList<>();
	
	public RelationPath(List<Integer> path) {
		super();
		this.path = path;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((path == null) ? 0 : path.hashCode());
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
		RelationPath other = (RelationPath) obj;
		if (path == null) {
			if (other.path != null)
				return false;
		} else if (!path.equals(other.path))
			return false;
		return true;
	}

	public int size() {
		return path.size();
	}
	
	public int get(int j) {
		return path.get(j);
	}
	
	@Override
	public String toString() {
		return "RelationPath [path=" + path + "]";
	}

	public static void main(String[] args){
		List<Integer> p1 = Arrays.asList(1,2,3);//new ArrayList<Integer>(A)
		RelationPath path1 = new RelationPath(p1);
		List<Integer> p2 = Arrays.asList(1,2,3);
		RelationPath path2 = new RelationPath(p2);
		System.out.println(path1);
	}

	

	
}
