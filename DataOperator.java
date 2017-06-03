package model;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Logger;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.structure.Graph;
import org.apache.tinkerpop.gremlin.structure.T;
import org.apache.tinkerpop.gremlin.structure.Vertex;
import org.apache.tinkerpop.gremlin.tinkergraph.structure.TinkerGraph;
import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;

public class DataOperator {
	// FB15k
//	static String TRAINING_FB15K = "data//FB15k//freebase_mtr100_mte100-train.txt";
//	static String VALID_FB15K = "data//FB15k//freebase_mtr100_mte100-valid.txt";
//	static String TEST_FB15K = "data//FB15k//freebase_mtr100_mte100-test.txt";
//	static String ENTITY2ID_FB15K = "data//FB15k//entity2id_thu.txt";
//	static String RELATION2ID_FB15K = "data//FB15k//relation2id.txt"; // NULL - 0
	
	// WN18
//	static String TRAINING_WN18 = "data//WN18//wordnet-mlj12-train.txt";
//	static String VALID_WN18 = "data//WN18//wordnet-mlj12-valid.txt";
//	static String TEST_WN18 = "data//WN18//wordnet-mlj12-test.txt";
//	static String ENTITY2ID_WN18 = "data//WN18//entity2id_thu.txt";
//	static String RELATION2ID_WN18 = "data//WN18//relation2id_thu.txt";

	String TRAINING;
	String VALID;
	String TEST;
	String ENTITY2ID;
	String RELATION2ID;
	
	public int entityNum = 0;
	public int relationNum = 0;
	public int trainingDataNum = 0;
	public int validDataNum = 0;
	public int testDataNum = 0;

	Map<String, Integer> entity2id = new HashMap<>();
	Map<Integer, String> id2entity = new HashMap<>();

	Map<String, Integer> relation2id = new HashMap<>();
	Map<Integer, String> id2relation = new HashMap<>();

	Map<Integer, Map<Integer, Integer>> leftEntityNum = new HashMap<>(); // relation - head - # of occurance
	Map<Integer, Map<Integer, Integer>> rightEntityNum = new HashMap<>(); // relation - tail - # of occurance

	double[] hpt; // head per tail
	double[] tph; // tail per head

	public List<int[]> trainingData = new ArrayList<>();
	Set<Triple> trainingDataObjSet = new HashSet<>();
	
	public List<int[]> validData = new ArrayList<>();
	Set<Triple> validDataObjSet = new HashSet<>();
	
	public List<int[]> testData = new ArrayList<>();
	Set<Triple> testDataObjSet = new HashSet<>();

	Set<Pair> trainPairSet = new HashSet<>();
	Set<Pair> validPairSet = new HashSet<>();
	Set<Pair> testPairSet = new HashSet<>();

	Graph graph = TinkerGraph.open();
	GraphTraversalSource graphTS = graph.traversal();
	Map<Integer, Vertex> vertexMap = new HashMap<>();

	PathContext pathMap = new PathContext();
	NeighborContext neighborMap = new NeighborContext();

	Logger logger = Logger.getLogger(this.getClass().getName());

	public DataOperator() {
		super();
		BasicConfigurator.configure();
	}
	
	public DataOperator(String dataSet) throws IOException {
		super();
		BasicConfigurator.configure();
		this.TRAINING = "data/" + dataSet + "/train.txt";
		this.VALID = "data/" + dataSet + "/valid.txt";
		this.TEST = "data/" + dataSet + "/test.txt";
		this.ENTITY2ID = "data/" + dataSet + "/entity2id.txt";
		this.RELATION2ID = "data/" + dataSet + "/relation2id.txt";
		readAllData();
		constructGraph();
	}

	/**
	 * update leftEntityNum and rightEntityNum
	 * @param h
	 * @param r
	 * @param t
	 */
	void updateMeanEntityNum(int headId, int relationId, int tailId){
		// update leftEntityNum
		if (leftEntityNum.containsKey(relationId)) {
			if (leftEntityNum.get(relationId).containsKey(headId)) {
				leftEntityNum.get(relationId).put(headId, 1 + leftEntityNum.get(relationId).get(headId));
			} else {
				leftEntityNum.get(relationId).put(headId, 1);
			}
		} else {
			Map<Integer, Integer> map = new HashMap<>();
			map.put(headId, 1);
			leftEntityNum.put(relationId, map);
		}

		// update rightEntityNum
		if (rightEntityNum.containsKey(relationId)) {
			if (rightEntityNum.get(relationId).containsKey(tailId)) {
				rightEntityNum.get(relationId).put(tailId, 1 + rightEntityNum.get(relationId).get(tailId));
			} else {
				rightEntityNum.get(relationId).put(tailId, 1);
			}
		} else {
			Map<Integer, Integer> map = new HashMap<>();
			map.put(tailId, 1);
			rightEntityNum.put(relationId, map);
		}
	}
	
	/**
	 * Read training data from file
	 * 
	 * @param filePath
	 * @throws IOException
	 */
	void readTrainingData(String filePath) throws IOException {
		assert filePath != null;
		BufferedReader reader = new BufferedReader(new FileReader(new File(filePath)));
		logger.info("Reading training data...");
		String line = null;
		while ((line = reader.readLine()) != null) {
			line = line.trim();
			String[] triple = line.split("\t");
			String head = triple[0].trim();
			String relation = triple[1].trim();
			String tail = triple[2].trim();
			int headId = entity2id.get(head);
			int relationId = relation2id.get(relation);
			int tailId = entity2id.get(tail);

			trainPairSet.add(new Pair(headId, tailId));

			// update leftEntityNum and rightEntityNum
			updateMeanEntityNum(headId, relationId, tailId);

			trainingData.add(new int[] { headId, relationId, tailId });
			trainingDataObjSet.add(new Triple(headId, relationId, tailId));
			trainingDataNum++;
		}

		logger.info("num of entities: " + entityNum);
		logger.info("num of relations: " + relationNum);
		logger.info("num of training data: " + trainingDataNum);

		reader.close();
	}
	
	void readValidData(String filePath) throws IOException {
		assert filePath != null;
		BufferedReader reader = new BufferedReader(new FileReader(new File(filePath)));
		logger.info("Reading valid data...");
		String line = null;
		while ((line = reader.readLine()) != null) {
			line = line.trim();
			String[] triple = line.split("\t");
			String head = triple[0].trim();
			String relation = triple[1].trim();
			String tail = triple[2].trim();
			int headId = entity2id.get(head);
			int relationId = relation2id.get(relation);
			int tailId = entity2id.get(tail);

			validPairSet.add(new Pair(entity2id.get(head), entity2id.get(tail)));

			if (entity2id.containsKey(head) && relation2id.containsKey(relation) && entity2id.containsKey(tail)){
				// update leftEntityNum and rightEntityNum
				//  updateMeanEntityNum(headId, relationId, tailId);

				// add current data
				validData.add(new int[] { entity2id.get(head), relation2id.get(relation), entity2id.get(tail) });
				validDataObjSet.add(new Triple(entity2id.get(head), relation2id.get(relation), entity2id.get(tail)));
				validDataNum++;
			}	
			else{
				logger.error("Some entity/relation in valid data is not in training data!");
				System.exit(1);
			}
				
		}
		logger.info("num of valid data: " + validData.size());
		reader.close();
	}

	void readTestData(String filePath) throws IOException {
		assert filePath != null;
		BufferedReader reader = new BufferedReader(new FileReader(new File(filePath)));
		logger.info("Reading test data...");
		String line = null;
		while ((line = reader.readLine()) != null) {
			line = line.trim();
			String[] triple = line.split("\t");
			String head = triple[0].trim();
			String relation = triple[1].trim();
			String tail = triple[2].trim();
			int headId = entity2id.get(head);
			int relationId = relation2id.get(relation);
			int tailId = entity2id.get(tail);

			testPairSet.add(new Pair(entity2id.get(head), entity2id.get(tail)));

			if (entity2id.containsKey(head) && relation2id.containsKey(relation) && entity2id.containsKey(tail)){
				// update leftEntityNum and rightEntityNum
				// updateMeanEntityNum(headId, relationId, tailId);

				// add current data
				testData.add(new int[] { entity2id.get(head), relation2id.get(relation), entity2id.get(tail) });
				testDataObjSet.add(new Triple(entity2id.get(head), relation2id.get(relation), entity2id.get(tail)));
				testDataNum++;
			}
			else{
				logger.error("Some entity/relation in test data is not in training data!");
			}
				
		}
		logger.info("num of test data: " + testData.size());
		reader.close();
	}
	
	void stats() {
		// calculate hpt and tph
		hpt = new double[relationNum];
		tph = new double[relationNum];

		for (int i = 1; i < relationNum; i++) {
			double numOfOccurance = 0;

			for (int j : leftEntityNum.get(i).keySet()) {
				numOfOccurance += leftEntityNum.get(i).get(j);
			}
			hpt[i] = numOfOccurance / leftEntityNum.get(i).keySet().size();

			numOfOccurance = 0;
			for (int j : rightEntityNum.get(i).keySet()) {
				numOfOccurance += rightEntityNum.get(i).get(j);
			}
			tph[i] = numOfOccurance / rightEntityNum.get(i).keySet().size();
		}
	}
	
	int relationCategory(int relationId){
		if(hpt[relationId]<1.5 && tph[relationId]<1.5)
			return 0; // 1-To-1
		else if(hpt[relationId]<1.5 && tph[relationId]>=1.5)
			return 1; // 1-To-n
		else if(hpt[relationId]>=1.5 && tph[relationId]<1.5)
			return 2; // n-To-1
		else //if(hpt[relationId]>=1.5 && tph[relationId]>=1.5) // n-To-n
			return 3;
	}
	
	public void constructGraph() {
		logger.info("start construct graph...");
		vertexMap = new HashMap<>();
		for (Entry<String, Integer> e : entity2id.entrySet()) {
			Vertex v = graph.addVertex(T.id, e.getValue(), T.label, String.valueOf(e.getValue()));
			vertexMap.put(e.getValue(), v);
		}
		for (int[] triple : trainingData) {
			vertexMap.get(triple[0]).addEdge(String.valueOf(triple[1]), vertexMap.get(triple[2]));
			// add reverse edge
			vertexMap.get(triple[2]).addEdge(String.valueOf(0 - triple[1]), vertexMap.get(triple[0]));
		}
		logger.info("graph constructed!");
	}

	/**
	 * Read entity - index from file
	 * 
	 * @param filePath
	 * @throws IOException
	 */
	void readEntityIdFromFile(String filePath) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(new File(filePath)));
		String line = "";
		while ((line = reader.readLine()) != null) {
			String[] array = line.split("\t");
			String entityName = array[0].trim();
			int entityId = Integer.valueOf(array[1].trim());
			entity2id.put(entityName, entityId);
			id2entity.put(entityId, entityName);
			entityNum++;
		}
		reader.close();
	}

	/**
	 * Read relation - index from file
	 * 
	 * @param filePath
	 * @throws IOException
	 */
	void readRelationIdFromFile(String filePath) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(new File(filePath)));
		String line = "";
		while ((line = reader.readLine()) != null) {
			String[] array = line.split("\t");
			String relationName = array[0].trim();
			int relationId = Integer.valueOf(array[1].trim());
			relation2id.put(relationName, relationId);
			id2relation.put(relationId, relationName);
			relationNum++;
		}
		reader.close();
	}

	

	public void readAllData() throws IOException {
		logger.info("Reading all data...");
		readEntityIdFromFile(ENTITY2ID);
		readRelationIdFromFile(RELATION2ID);
		readTrainingData(TRAINING);
		readValidData(VALID);
		readTestData(TEST);
		stats(); // get hpt and tph
		
	}

	public boolean containTriple(Triple triple) {
		if(trainingDataObjSet.contains(triple) || validDataObjSet.contains(triple) || testDataObjSet.contains(triple))
			return true;
		else
			return false;
	}
	
	public boolean containPair(Pair pair) {
		if(trainPairSet.contains(pair) || validPairSet.contains(pair) || testPairSet.contains(pair))
			return true;
		else
			return false;
	}
	
	/*
	public NeighborMap sampleNeighborsOfAllEntities(int n){
		NeighborMap neighborMap = new NeighborMap();
		for(int i=0; i<entityNum; i++){
			logger.info((i+1) + "/" + entityNum);
			Set<Neighbor> sampleSet = new HashSet<>();
			List<Triple> allNeighbors = findNeighbors(i);
			if(allNeighbors.size() <= n){
				for(Triple t : allNeighbors){
					sampleSet.add(new Neighbor(t.relation, t.tail));
				}
			}
			else{
				while(sampleSet.size()<n){
					Triple t = allNeighbors.get((int) (Math.random()*allNeighbors.size()));
					sampleSet.add(new Neighbor(t.relation, t.tail));
				}
			}
			neighborMap.put(i, sampleSet);
		}
		return neighborMap;
	}
	
	void writeNeighborMapToFile(NeighborMap neighborMap, String name) throws IOException{
		logger.info("writing NeighborMap to " + name);
		Gson gson = new Gson();
		FileWriter writer = new FileWriter(name);
		gson.toJson(neighborMap, writer);
		writer.close();
	}
	
	List<RelationPath> findPath2(int startEntityId, int endEntityId) {
		Vertex startVertex = vertexMap.get(startEntityId);
		Vertex endVertex = vertexMap.get(endEntityId);

		List<RelationPath> pathList = new ArrayList<>();

		// find 2-step path
		Iterator<Path> itr = graphTS.V(startVertex).outE().inV().outE().inV().hasId(endEntityId).simplePath().path();
		while (itr.hasNext()) {
			Path path = itr.next();
			pathList.add(new RelationPath(new ArrayList<>(Arrays.asList(Integer.valueOf(((Edge) path.get(1)).label()),
					Integer.valueOf(((Edge) path.get(3)).label())))));
		}

		return pathList;
	}

	List<RelationPath> findPath2(String startEntity, String endEntity) {
		int startEntityId = entity2id.get(startEntity);
		int endEntityId = entity2id.get(endEntity);

		return findPath2(startEntityId, endEntityId);
	}

	List<RelationPath> findPath2(int startEntityId, int endEntityId, String logInfo) {
		logger.info(logInfo);
		return findPath2(startEntityId, endEntityId);
	}

	
	List<RelationPath> findPath3(int startEntityId, int endEntityId) {
		Vertex startVertex = vertexMap.get(startEntityId);
		Vertex endVertex = vertexMap.get(endEntityId);

		List<RelationPath> pathList = new ArrayList<>();

		// find 3-step path
		Iterator<Path> itr = graphTS.V(startVertex).outE().inV().outE().inV().outE().inV().hasId(endEntityId)
				.simplePath().path();
		while (itr.hasNext()) {
			Path path = itr.next();
			pathList.add(new RelationPath(new ArrayList<>(Arrays.asList(Integer.valueOf(((Edge) path.get(1)).label()),
					Integer.valueOf(((Edge) path.get(3)).label()), Integer.valueOf(((Edge) path.get(5)).label())))));
		}

		return pathList;
	}

	List<RelationPath> findPath3(String startEntity, String endEntity) {
		int startEntityId = entity2id.get(startEntity);
		int endEntityId = entity2id.get(endEntity);

		return findPath3(startEntityId, endEntityId);
	}

	List<RelationPath> findPath3(int startEntityId, int endEntityId, String logInfo) {
		logger.info(logInfo);
		return findPath3(startEntityId, endEntityId);
	}

	void findAll2Path(Collection<Pair> collection) {
		ExecutorService fixedThreadPool = Executors.newFixedThreadPool(5);
		Iterator<Pair> itr = collection.iterator();
		int i = 0;
		while (itr.hasNext()) {
			Pair pair = itr.next();
			int headId = pair.headId;
			int tailId = pair.tailId;
			String logInfo = "finding 2 step path of data " + (i + 1) + "/" + collection.size();
			if (!this.pathMap.containsKey(pair.toString())) {
				fixedThreadPool.execute(new Runnable() {
					@Override
					public void run() {
						pathMap.put(pair.toString(), findPath2(pair.headId, pair.tailId, logInfo));
					}
				});
			}
			i++;
		}
		fixedThreadPool.shutdown();
		try {
			while (!fixedThreadPool.awaitTermination(10, TimeUnit.SECONDS))
				;
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	void findAll3Path(Collection<Pair> collection) {
		ExecutorService fixedThreadPool = Executors.newFixedThreadPool(5);
		Iterator<Pair> itr = collection.iterator();
		int i = 0;
		while (itr.hasNext()) {
			Pair pair = itr.next();
			int headId = pair.headId;
			int tailId = pair.tailId;
			String logInfo = "finding 3 step path of data " + (i + 1) + "/" + collection.size();
			if (!pathMap.containsKey(pair.toString())) {
				fixedThreadPool.execute(new Runnable() {
					@Override
					public void run() {
						pathMap.put(pair.toString(), findPath3(pair.headId, pair.tailId, logInfo));
					}
				});
			}
			i++;
		}
		fixedThreadPool.shutdown();
		try {
			while (!fixedThreadPool.awaitTermination(10, TimeUnit.SECONDS))
				;
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	

	void writePathMapToFile(PathMap pathMap, String name) throws IOException {
		logger.info("writing PathMap to " + name);
		Gson gson = new Gson();
		FileWriter writer = new FileWriter(name);
		gson.toJson(pathMap, writer);
		writer.close();
	}

	PathMap samplePathMap(String path, int n) throws IOException {
		PathMap pathmap = readPathMapFromFile(path);
		PathMap newPathMap = new PathMap();
		for(Entry<String, List<RelationPath>> e : pathmap.entrySet()) {
			String key = e.getKey();
			List<RelationPath> value = e.getValue();
			if(e.getValue().size() > n) {
				List<RelationPath> sampleList = new ArrayList<>();
//				Set<Integer> indexSet = new HashSet<>();
//				while(indexSet.size() < n) {
//					indexSet.add((int) (value.size()*Math.random()));
//				}
//				for(int index : indexSet){
//					sampleList.add(value.get(index));
//				}
				while(sampleList.size() < n)
					sampleList.add(value.get( (int) (value.size()*Math.random()) ));
				newPathMap.put(key, sampleList);
			}
			else {
				newPathMap.put(key, value);
			}
		}
		return newPathMap;
	}
	*/
	
	NeighborContext readNeighborFromFile(String name) throws IOException {
		logger.info("reading NeighborMap from " + name);
		Gson gson = new Gson();
		JsonReader reader = new JsonReader(new BufferedReader(new FileReader(name)));
		NeighborContext map = gson.fromJson(reader, NeighborContext.class);
		reader.close();
		return map;
	}
	
//	void writeNeighborToFile(NeighborMap neighborMap, String name) throws IOException {
//		logger.info("writing NeighborMap to " + name);
//		Gson gson = new Gson();
//		FileWriter writer = new FileWriter(name);
//		gson.toJson(neighborMap, writer);
//		writer.close();
//	}
	
	PathContext readPathMapFromFile(String name) throws IOException {
		logger.info("reading PathMap from " + name);
		Gson gson = new Gson();
		JsonReader reader = new JsonReader(new BufferedReader(new FileReader(name)));
		PathContext map = gson.fromJson(reader, PathContext.class);
		reader.close();
		return map;
	}
	
	<T> void writeObjToFile(T obj, String name) throws IOException {
		logger.info("writing "+obj.getClass()+" to " + name);
		Gson gson = new Gson();
		FileWriter writer = new FileWriter(name);
		gson.toJson(obj, writer);
		writer.close();
	}
	
	void generateId() {
		logger.info("generating id for entities and relations...");
		
	}
	
	public void prepare(String pathMapPath, String neighborMapPath) throws IOException{
		this.pathMap = readPathMapFromFile(pathMapPath);
		this.neighborMap = readNeighborFromFile(neighborMapPath);
	}
	
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		DataOperator dataOperator = new DataOperator();
		NeighborContext neighborMap = dataOperator.readNeighborFromFile("neighborMap_fb15k_sample10.json");
		for(int key : neighborMap.neighborMap.keySet()){
			Set<Neighbor> neighborSet = neighborMap.get(key);
			for(Neighbor neighbor : neighborSet){
				neighbor.r++;
			}
		}
		dataOperator.writeObjToFile(neighborMap, "newNeighborMap.json");
		
		// convert
		/*
		PathMap map_train = new PathMap();
		PathMap map_test = new PathMap();
		PathMap map_com = new PathMap();
		for(String key : map_oldPathMap_train.pathMap.keySet()){
			List<RelationPath> list = new ArrayList<>();
			for(List<Integer> path : map_oldPathMap_train.get(key))
				list.add(new RelationPath(path));
			map_train.put(key, list);
		}
		for(String key : map_oldPathMap_test.pathMap.keySet()){
			List<RelationPath> list = new ArrayList<>();
			for(List<Integer> path : map_oldPathMap_test.get(key))
				list.add(new RelationPath(path));
			map_test.put(key, list);
		}
		map_com.putAll(map_train);
		map_com.putAll(map_test);
		dataOperator.writePathMapToFile(map_train, "path_fb15k_sample_10_train.json");
		dataOperator.writePathMapToFile(map_test, "path_fb15k_sample_10_test.json");
		dataOperator.writePathMapToFile(map_com, "path_fb15k_sample_10_com.json");
		*/
		
//		DataOperator dataOperator = new DataOperator();
//		dataOperator.readAllData("wn18");
//		dataOperator.constructGraph();
//		NeighborMap neighborMap = dataOperator.sampleNeighborsOfAllEntities(10);
//		int j=0;
//		for(int i=0; i<dataOperator.entityNum; i++){
//			if(neighborMap.get(i).size()==0)
//				j++;
//		}
//		System.out.println(j);
//		dataOperator.writeNeighborMapToFile(neighborMap, "neighborMap_wn18_sample_10.json");
		
//		List<Pair> pairList = new ArrayList<>(dataOperator.testPairSet);
//		dataOperator.findAll2Path(pairList.subList(50000, 57360));
//		dataOperator.writePathMapToFile(dataOperator.pathMap, "test_2PathMap_50000_57360.json");
		
		//sample
//		dataOperator.pathMap = dataOperator.readPathMapFromFile("train_2PathMap_all.json");
//		PathMap sampledPathMap = dataOperator.samplePathMap("test_2and3PathMap_all.json", 10);
//		dataOperator.writePathMapToFile(sampledPathMap, "test_2and3PathMap_sample_10.json");
//		dataOperator.test();
//		dataOperator.combine();
//		dataOperator.stats();

	}
	
	void combine() throws IOException {
		// combine
		PathContext map1 = readPathMapFromFile("test_2PathMap_all.json");
		PathContext map2 = readPathMapFromFile("test_3PathMap_all.json");
//		PathMap map3 = readPathMapFromFile("test_2PathMap_20000_30000.json");
//		PathMap map4 = readPathMapFromFile("test_2PathMap_30000_40000.json");
//		PathMap map5 = readPathMapFromFile("test_2PathMap_40000_50000.json");
//		PathMap map6 = readPathMapFromFile("test_2PathMap_50000_57360.json");
//		PathMap map7 = readPathMapFromFile("train_3PathMap_sample_10_7.json");
//		PathMap map8 = readPathMapFromFile("train_3PathMap_sample_10_8.json");
		PathContext all = new PathContext();
		all.putAll(map1);
		all.putAll(map2);
//		all.putAll(map3);
//		all.putAll(map4);
//		all.putAll(map5);
//		all.putAll(map6);
//		all.putAll(map7);
//		all.putAll(map8);
//		writePathMapToFile(all, "test_2and3PathMap_all.json");
		
	}
	
	void getStats() throws IOException {
		PathContext all = readPathMapFromFile("test_3PathMap_all.json");
		System.out.println("path map size: " + all.pathMap.size());
		int min_num = Integer.MAX_VALUE;
		int max_num = Integer.MIN_VALUE;
		int num_0 = 0;
		int num_1_10 = 0;
		int num_10_20 = 0;
		int num_20_30 = 0;
		int num_30_40 = 0;
		int num_40_50 = 0;
		int num_50_60 = 0;
		int num_60_70 = 0;
		int num_70_80 = 0;
		int num_80_90 = 0;
		int num_90_100 = 0;
		int num_100more = 0;
		int sum = 0;
		Set<String> num0PairSet = new HashSet<>();
		for(String key : all.pathMap.keySet()){
			int pathNum = all.pathMap.get(key).size();
			sum += pathNum;
			if(pathNum < min_num)
				min_num = pathNum;
			if(pathNum > max_num)
				max_num = pathNum;
			if(pathNum == 0){
				num0PairSet.add(key);
				num_0++;
			}
			else if(pathNum < 10)
				num_1_10++;
			else if(pathNum < 20)
				num_10_20++;
			else if(pathNum < 30)
				num_20_30++;
			else if(pathNum < 40)
				num_30_40++;
			else if(pathNum < 50)
				num_40_50++;
			else if(pathNum < 60)
				num_50_60++;
			else if(pathNum < 70)
				num_60_70++;
			else if(pathNum < 80)
				num_70_80++;
			else if(pathNum < 90)
				num_80_90++;
			else if(pathNum < 100)
				num_90_100++;
			else// if(pathNum < 20)
				num_100more++;
			
		}
		
		System.out.println("min num: " + min_num);
		System.out.println("max num: " + max_num);
		System.out.println("sum: " + sum);
		System.out.println("avg size: " + (double)sum/all.pathMap.size());
		System.out.println("0: " + num_0);
		System.out.println("1-10: " + num_1_10);
		System.out.println("10-20: " + num_10_20);
		System.out.println("20-30: " + num_20_30);
		System.out.println("30-40: " + num_30_40);
		System.out.println("40-50: " + num_40_50);
		System.out.println("50-60: " + num_50_60);
		System.out.println("60-70: " + num_60_70);
		System.out.println("70-80: " + num_70_80);
		System.out.println("80-90: " + num_80_90);
		System.out.println("90-100: " + num_90_100);
		System.out.println("100+: " + num_100more);
		
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File("test_3_num0_pairSet.txt")));
		for(String s : num0PairSet){
			writer.write(s + "\n");
		}
		writer.close();
		
	}
}
