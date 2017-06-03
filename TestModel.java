package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.log4j.Logger;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class TestModel {
	String dataset;
	int n;
	int threadNum;
	String pathContextPath;
	String neighborContextPath;
	String entity2VecPath;
	String relation2VecPath;
	
	DataOperator dataOperator;
	
	double[][] entityVec;
	double[][] relationVec;
	
	AtomicInteger h_sum = new AtomicInteger();
	AtomicInteger t_sum = new AtomicInteger();
	AtomicInteger h_sum_filter = new AtomicInteger();
	AtomicInteger t_sum_filter = new AtomicInteger();
	AtomicInteger h_hit10 = new AtomicInteger();
	AtomicInteger h_hit10_filter = new AtomicInteger();
	AtomicInteger t_hit10 = new AtomicInteger();
	AtomicInteger t_hit10_filter = new AtomicInteger();
	
	AtomicInteger h_hit10_1_1 = new AtomicInteger();
	AtomicInteger t_hit10_1_1 = new AtomicInteger();
	AtomicInteger h_hit10_1_n = new AtomicInteger();
	AtomicInteger t_hit10_1_n = new AtomicInteger();
	AtomicInteger h_hit10_n_1 = new AtomicInteger();
	AtomicInteger t_hit10_n_1 = new AtomicInteger();
	AtomicInteger h_hit10_n_n = new AtomicInteger();
	AtomicInteger t_hit10_n_n = new AtomicInteger();
	
	AtomicInteger num_1_1 = new AtomicInteger();
	AtomicInteger num_1_n = new AtomicInteger();
	AtomicInteger num_n_1 = new AtomicInteger();
	AtomicInteger num_n_n = new AtomicInteger();
	
//	AtomicInteger lp_1 = new AtomicInteger();
//	AtomicInteger lp_1_filter = new AtomicInteger();
//	AtomicInteger rp_1 = new AtomicInteger();
//	AtomicInteger rp_1_filter = new AtomicInteger();
//	AtomicInteger lp_3 = new AtomicInteger();
//	AtomicInteger lp_3_filter = new AtomicInteger();
//	AtomicInteger rp_3 = new AtomicInteger();
//	AtomicInteger rp_3_filter = new AtomicInteger();
	
	Logger logger = Logger.getLogger(this.getClass().getName());
	
	public TestModel() {
		super();
//		BasicConfigurator.configure();
	}

	public TestModel(String dataset, int n, int threadNum, String pathContextPath, String neighborContextPath, String entity2VecPath, String relation2VecPath) throws IOException {
		super();
		this.dataset = dataset;
		this.n = n;
		this.threadNum = threadNum;
		this.pathContextPath = pathContextPath;
		this.neighborContextPath = neighborContextPath;
		this.entity2VecPath = entity2VecPath;
		this.relation2VecPath = relation2VecPath;
		
		this.dataOperator = new DataOperator(dataset);
		this.dataOperator.prepare(this.pathContextPath, this.neighborContextPath);
		loadVectors(this.entity2VecPath, this.relation2VecPath);
	}

	void readEntity2Vec(String filePath) throws IOException {
		this.entityVec = new double[dataOperator.entityNum][n];
		BufferedReader reader = new BufferedReader(new FileReader(new File(filePath)));
		String line = "";
		int i = 0;
		while((line=reader.readLine()) != null){
			line = line.trim();
			String[] a = line.split("\t");
			if(a.length != this.n){
				logger.error("entity vector size is not equal to " + n);
				System.exit(1);
			}
			for(int j=0; j<this.n; j++){
				entityVec[i][j] = Double.valueOf(a[j]);
			}
			i++;
		}
		reader.close();
	}
	
	void readRelation2Vec(String filePath) throws IOException {
		this.relationVec = new double[dataOperator.relationNum][n];
		BufferedReader reader = new BufferedReader(new FileReader(new File(filePath)));
		String line = "";
		int i = 0;
		while((line=reader.readLine()) != null){
			line = line.trim();
			String[] a = line.split("\t");
			if(a.length != this.n){
				logger.error("relation vector size is not equal to" + n);
				System.exit(1);
			}
			for(int j=0; j<this.n; j++){
				relationVec[i][j] = Double.valueOf(a[j]);
			}
			i++;
		}
		reader.close();
	}
	
	/**
	 * TransE score of a triple
	 * 
	 * @param h
	 * @param r
	 * @param t
	 * @return
	 */
	double score_TransE(int h, int r, int t) {
		double sum = 0;
		if (r > 0) {
			for (int i = 0; i < n; i++)
				sum += Math.abs(entityVec[h][i] + relationVec[r][i] - entityVec[t][i]);
		}
		else {
			for (int i = 0; i < n; i++)
				sum += Math.abs(entityVec[t][i] + relationVec[0-r][i] - entityVec[h][i]);
		}
		
		return sum;
	}

	double score_path(int h, RelationPath p, int t) {
		double sum = 0;
		double[] pSum = new double[n];
		for (int i = 0; i < n; i++) {
			pSum[i] = 0;
			for (int j = 0; j < p.size(); j++) {
				if(p.get(j) > 0)
					pSum[i] += relationVec[p.get(j)][i];
				else
					pSum[i] -= relationVec[0-p.get(j)][i];
			}

		}
		for (int i = 0; i < n; i++) {
			sum += Math.abs(entityVec[h][i] + pSum[i] - entityVec[t][i]);
		}
		return sum;
	}
	
	double score_context(int h, int r, int t){
		return ln_sigmoid(f1(h)) + ln_sigmoid(f2(h,t)) + ln_sigmoid(f3(h,r,t));
	}
	
	public double sigmoid(double z) {
		return (double) 1 / (1 + Math.exp(-z));
	}
	
	public double ln_sigmoid(double z) {
		return Math.log(sigmoid(z));
	}
	
	double f1(int h) {
		Set<Neighbor> neighbors = dataOperator.neighborMap.get(h);
		// if there are no neighbors
		if(neighbors == null || neighbors.size() == 0)
			return -100;
		double sum = 0;
		for (Neighbor neighbor : neighbors) {
			sum += score_TransE(h, neighbor.r, neighbor.t);
		}
		return (0 - sum) / neighbors.size();
	}

	double f2(int h, int t) {
		double sum = 0;
		List<RelationPath> paths = dataOperator.pathMap.get(h + " " + t);
		// if does not exist paths between two nodes
		if(paths == null || paths.size() == 0)
			return -100;
		for (int i = 0; i < paths.size(); i++) {
			sum += score_path(h, paths.get(i), t);
		}
		return (0 - sum) / paths.size();
	}

	double f3(int h, int r, int t) {
		return 0 - score_TransE(h, r, t);
	}
	
	void loadVectors(String entity2VecPath, String relation2VecPath) throws IOException{
		// initialize vectors of entities and relations
		readEntity2Vec(entity2VecPath);
		readRelation2Vec(relation2VecPath);
	}
	
//	void prepare() throws IOException {
//		
//	}
	
	@SuppressWarnings("unchecked")
	void test(int threadNum) {
		ExecutorService fixedThreadPool = Executors.newFixedThreadPool(threadNum);
		// iterate over all test triples
		for(int i=0; i<dataOperator.testDataNum; i++){
			
			int h = dataOperator.testData.get(i)[0];
			int r = dataOperator.testData.get(i)[1];
			int t = dataOperator.testData.get(i)[2];
			
			int rCategory = dataOperator.relationCategory(r); // get category of current relation
		    
			if(rCategory==0)
				num_1_1.set(num_1_1.incrementAndGet());
			else if(rCategory==1)
				num_1_n.set(num_1_n.incrementAndGet());
			else if(rCategory==2)
				num_n_1.set(num_n_1.incrementAndGet());
			else // rCategory==3
				num_n_n.set(num_n_n.incrementAndGet());
			
			String logInfo = (i+1)+"/"+dataOperator.testDataNum;
			fixedThreadPool.execute(new Runnable() {
				@Override
				public void run() {
					logger.info(logInfo);
					
					List<Tuple> sortList = new ArrayList<>();
					
					// predicting head
					for(int j=0; j<dataOperator.entityNum; j++){
						double score_tmp = score_context(j, r, t);
						
						sortList.add(new Tuple(j, score_tmp));
					}
					Collections.sort(sortList); // sort by ascending order
					int filter = 0;
					// from back to front
					for(int k=sortList.size()-1; k>=0; k--){
						Triple triple = new Triple(sortList.get(k).entityId, r, t);
						if(!dataOperator.containTriple(triple))
							filter++;
						
						if(sortList.get(k).entityId == h){
							h_sum.set(h_sum.addAndGet(sortList.size()-k));
							h_sum_filter.set(h_sum_filter.addAndGet(filter+1));
							
							if(sortList.size()-k <= 10){
								h_hit10.set(h_hit10.incrementAndGet());
							}
							if(filter < 10) {
								if(rCategory==0)
									h_hit10_1_1.set(h_hit10_1_1.incrementAndGet());
								else if(rCategory==1)
									h_hit10_1_n.set(h_hit10_1_n.incrementAndGet());
								else if(rCategory==2)
									h_hit10_n_1.set(h_hit10_n_1.incrementAndGet());
								else // rCategory==3
									h_hit10_n_n.set(h_hit10_n_n.incrementAndGet());
								
								h_hit10_filter.set(h_hit10_filter.incrementAndGet());
							}
								
							break;
						}
						
					}
					sortList.clear();
					
					// predicting tail
					for(int j=0; j<dataOperator.entityNum; j++){
						double score_tmp = score_context(h, r, j);
						
						sortList.add(new Tuple(j, score_tmp));
					}
					Collections.sort(sortList); // sort by ascending order
					
					filter = 0;
					// from back to front
					for(int k=sortList.size()-1; k>=0; k--){
						Triple triple = new Triple(h, r, sortList.get(k).entityId);
						if(!dataOperator.containTriple(triple))
							filter++;
						
						if(sortList.get(k).entityId == t){
							t_sum.set(t_sum.addAndGet(sortList.size()-k));
							t_sum_filter.set(t_sum_filter.addAndGet(filter+1));
							
							if(sortList.size()-k <= 10){
								t_hit10.set(t_hit10.incrementAndGet());
								
								
							}
							if(filter < 10){
								if(rCategory==0)
									t_hit10_1_1.set(t_hit10_1_1.incrementAndGet());
								else if(rCategory==1)
									t_hit10_1_n.set(t_hit10_1_n.incrementAndGet());
								else if(rCategory==2)
									t_hit10_n_1.set(t_hit10_n_1.incrementAndGet());
								else // rCategory==3
									t_hit10_n_n.set(t_hit10_n_n.incrementAndGet());
								
								t_hit10_filter.set(t_hit10_filter.incrementAndGet());
							}
								
							break;
						}
						
					}
				}
			});
			
			
		}
		// wait for threads finish
		fixedThreadPool.shutdown();
		try {
			while (!fixedThreadPool.awaitTermination(10, TimeUnit.SECONDS))
				;
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		int testNum = dataOperator.testDataNum;
		System.out.println("MeanRank(Raw)\tMeanRank(Filter)\tHit@10(Raw)\tHit@10(Filter)");
		System.out.println("left: "+h_sum.doubleValue()/testNum+"\t"+h_hit10.doubleValue()/testNum+"\t"+h_sum_filter.doubleValue()/testNum+"\t"+h_hit10_filter.doubleValue()/testNum);
		System.out.println("right: "+t_sum.doubleValue()/testNum+"\t"+t_hit10.doubleValue()/testNum+"\t"+t_sum_filter.doubleValue()/testNum+"\t"+t_hit10_filter.doubleValue()/testNum);
		System.out.println("avg: "+(h_sum.doubleValue()+t_sum.doubleValue())/testNum/2+"\t"+(h_sum_filter.doubleValue()+t_sum_filter.doubleValue())/testNum/2+"\t"+(h_hit10.doubleValue()+t_hit10.doubleValue())/testNum/2+"\t"+(h_hit10_filter.doubleValue()+t_hit10_filter.doubleValue())/testNum/2);
		System.out.println("head: " + h_hit10_1_1.doubleValue()/num_1_1.doubleValue() +" "+ h_hit10_1_n.doubleValue()/num_1_n.doubleValue() +" "+ h_hit10_n_1.doubleValue()/num_n_1.doubleValue() +" "+ h_hit10_n_n.doubleValue()/num_n_n.doubleValue());
		System.out.println("tail: " + t_hit10_1_1.doubleValue()/num_1_1.doubleValue() +" "+ t_hit10_1_n.doubleValue()/num_1_n.doubleValue() +" "+ t_hit10_n_1.doubleValue()/num_n_1.doubleValue() +" "+ t_hit10_n_n.doubleValue()/num_n_n.doubleValue());
	}
	
	class Tuple implements Comparable{
		int entityId;
		double score;
		
		public Tuple(int entityId, double score) {
			super();
			this.entityId = entityId;
			this.score = score;
		}

		@Override
		public int compareTo(Object o) {
			Tuple tuple = (Tuple) o;
			// ascending order
			if(score < tuple.score)
				return -1;
			if(score > tuple.score)
				return 1;
			return 0;
		}
		
	}
	
	public static void main(String[] args) throws IOException {
		JsonParser parser = new JsonParser();
		JsonObject config = parser.parse(new FileReader("testConfig.json")).getAsJsonObject();
		String dataset = config.get("dataset").getAsString();
		int n = config.get("n").getAsInt();
		int threadNum = config.get("threadNum").getAsInt();
		String pathContextPath = config.get("pathContextPath").getAsString();
		String neighborContextPath = config.get("neighborContextPath").getAsString();
		String entity2VecPath = config.get("entity2VecPath").getAsString();
		String relation2VecPath = config.get("relation2VecPath").getAsString();
		
		// args: 0-dataset 1-dimensions 2-threadNum 3-pathMap_path 4-neighborMap_path 5-entity2vecPath 6-relation2vecPath
		// java -jar testModel.jar fb15k 50 30 vec/entity2vec.50.999 vec/relation2vec.50.999
//		if(args.length != 7){
//			System.err.println("Wrong args size!");
//			return;
//		}
		
//		String dataset = args[0];
//		int n = Integer.valueOf(args[1]);
//		int threadNum = Integer.valueOf(args[2]);
//		String pathMap_path = args[3];
//		String neighborMap_path = args[4];
//		String entity2VecPath = args[5];
//		String relation2VecPath = args[6];		
		
		TestModel testModel = new TestModel(dataset, n, threadNum, pathContextPath, neighborContextPath, entity2VecPath, relation2VecPath);
//		testModel.prepare(dataset, pathMap_path, neighborMap_path, entity2VecPath, relation2VecPath);
		testModel.test(threadNum);
	}

}
