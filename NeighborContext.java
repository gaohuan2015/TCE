package model;

import java.util.HashMap;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class NeighborContext {
//	ConcurrentHashMap<Integer, Set<Neighbor>> neighborMap = new ConcurrentHashMap<>();
	HashMap<Integer, Set<Neighbor>> neighborMap = new HashMap<>();
	
	Set<Neighbor> get(int key){
		return this.neighborMap.get(key);
	}
	
	Set<Neighbor> put(int key, Set<Neighbor> value){
		return this.neighborMap.put(key, value);
	}
}
