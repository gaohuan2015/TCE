package model;

public class Triple {
	public int head;
	public int relation;
	public int tail;
	
	public Triple(int head, int relation, int tail) {
		super();
		this.head = head;
		this.relation = relation;
		this.tail = tail;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + head;
		result = prime * result + relation;
		result = prime * result + tail;
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
		Triple other = (Triple) obj;
		if (head != other.head)
			return false;
		if (relation != other.relation)
			return false;
		if (tail != other.tail)
			return false;
		return true;
	}

	
	
}
