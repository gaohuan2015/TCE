package model;

import java.util.List;
import java.util.Set;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;

public class PathContext {
	ConcurrentHashMap<String, List<RelationPath>> pathMap = new ConcurrentHashMap<>();
	
	public PathContext() {
		super();
		// TODO Auto-generated constructor stub
	}

	boolean containsKey(String key) {
		return this.pathMap.containsKey(key);
	}
	
	void put(String pair, List<RelationPath> pathList){
		this.pathMap.put(pair, pathList);
	}
	
	List<RelationPath> get(String key){
		return this.pathMap.get(key);
	}
	
	void putAll(PathContext pathMap) {
		this.pathMap.putAll(pathMap.pathMap);
	}
	
	Set<Entry<String, List<RelationPath>>> entrySet() {
		return this.pathMap.entrySet();
	}
}
