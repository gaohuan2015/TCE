package model;

public class Neighbor{
	int r;
	int t;
	
	public Neighbor(int r, int t) {
		super();
		this.r = r;
		this.t = t;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + r;
		result = prime * result + t;
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
		Neighbor other = (Neighbor) obj;
		if (r != other.r)
			return false;
		if (t != other.t)
			return false;
		return true;
	}
	
	
	
}
