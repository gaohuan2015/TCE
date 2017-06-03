package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.Set;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Logger;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class TrainModel {
	String dataset;
	int n; // dims of embedding
	double rate; // learning rate
	int epochNum;
	int batchNum;
	int threadNum;
	String neighborContextPath;
	String pathContextPath;
	String entity2VecPath;
	String relation2VecPath;
	String outDir;
	
	boolean init = true; // true - init vectors randomly, false - load trained vectors
	
	double[][] entityVec;
	double[][] relationVec;
	
	double loss;

	DataOperator dataOperator;

	Logger logger = Logger.getLogger(this.getClass().getName());
	

	public double sigmoid(double z) {
		return (double) 1 / (1 + Math.exp(-z));
	}
	
	public double ln_sigmoid(double z) {
		return Math.log(sigmoid(z));
	}

	/**
	 * Normalize a vector
	 * 
	 * @param vec
	 */
	void normalize(double[] vec) {
		double sum = 0;
		for (int i = 0; i < vec.length; i++) {
			sum += Math.pow(vec[i], 2);
		}
		for (int i = 0; i < vec.length; i++) {
			vec[i] /= Math.sqrt(sum);
		}
	}

	/**
	 * read entity vectors from file
	 * @param filePath
	 * @throws IOException
	 */
	void readEntity2Vec(String filePath) throws IOException {
		this.entityVec = new double[dataOperator.entityNum][n];
		BufferedReader reader = new BufferedReader(new FileReader(new File(filePath)));
		String line = "";
		int i = 0;
		while((line=reader.readLine()) != null){
			line = line.trim();
			String[] a = line.split("\t");
			if(a.length != this.n){
				logger.error("entity vector size is not equal to n");
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
		int i = 1; // 0 correspond to the NULL relation
		while((line=reader.readLine()) != null){
			line = line.trim();
			String[] a = line.split("\t");
			if(a.length != this.n){
				logger.error("relation vector size is not equal to n");
				System.exit(1);
			}
			for(int j=0; j<this.n; j++){
				relationVec[i][j] = Double.valueOf(a[j]);
			}
			i++;
		}
		reader.close();
	}
	
	void initVectors() {
		// initialize vectors of entities and relations
		this.entityVec = new double[dataOperator.entityNum][this.n];
		this.relationVec = new double[dataOperator.relationNum][this.n];

		// assign random values to entity and relation vectors and normalize them
		for (int i = 0; i < dataOperator.entityNum; i++) {
			for (int j = 0; j < n; j++) {
				entityVec[i][j] = (2 * Math.random() - 1.0) * 6 / Math.sqrt(n);
			}
			normalize(entityVec[i]);
		}

		for (int i = 1; i < dataOperator.relationNum; i++) {
			for (int j = 0; j < n; j++) {
				relationVec[i][j] = (2 * Math.random() - 1.0) * 6 / Math.sqrt(n);
			}
			normalize(relationVec[i]);
		}
	}
	
	void loadVectors(String entity2VecPath, String relation2VecPath) throws IOException{
		readEntity2Vec(entity2VecPath);
		readRelation2Vec(relation2VecPath);
	}
	
	void train() throws IOException {
		logger.info("start training model...");
		
		int batchSize = dataOperator.trainingDataNum / this.batchNum;
		logger.info("batch size: " + batchSize);
		
		// iterate over each epoch
		for (int epoch = 0; epoch < epochNum; epoch++) {
//			logger.info("epoch:" + epoch);
			loss = 0;
			// iterate over each batch
			for (int batch = 0; batch < batchNum; batch++) {
//				logger.info("epoch:" + epoch + " batch:" + batch);
//				entityVecTmp = entityVec.clone();
//				relationVecTmp = relationVec.clone();
				ConcurrentHashMap<Integer, List<Double>> entityUpdateMap = new ConcurrentHashMap<>();
				ConcurrentHashMap<Integer, List<Double>> relationUpdateMap = new ConcurrentHashMap<>();
				
				ExecutorService fixedThreadPool = Executors.newFixedThreadPool(this.threadNum);
				
				for (int k = 0; k < batchSize; k++) {
					fixedThreadPool.execute(new Runnable() {
						@Override
						public void run() {
							// TODO Auto-generated method stub
							int trainingDataIdx = (int) (Math.random() * dataOperator.trainingDataNum); // pick a training data randomly
							int h = dataOperator.trainingData.get(trainingDataIdx)[0];
							int r = dataOperator.trainingData.get(trainingDataIdx)[1];
							int t = dataOperator.trainingData.get(trainingDataIdx)[2];
							int h_n = h;
							int r_n = r;
							int t_n = t;

							double pr = dataOperator.tph[r] / (dataOperator.tph[r] + dataOperator.hpt[r]);
							if(Math.random() < pr){ // replace head
								h_n = (int) (Math.random() * dataOperator.entityNum); // pick a negative head
								while(dataOperator.containTriple(new Triple(h_n, r, t)))
									h_n = (int) (Math.random() * dataOperator.entityNum);
							}
							else{ // replace tail
								t_n = (int) (Math.random() * dataOperator.entityNum); // pick a negative tail
								while(dataOperator.containTriple(new Triple(h, r, t_n)))
									t_n = (int) (Math.random() * dataOperator.entityNum);
							}

							// gradient descent
							// gradientDescent(h, r, t, h_n, r_n, t_n);
							double f1 = f1(h);
							double f2 = f2(h, t);
							double f3 = f3(h, r, t);
							double f1_n = 0-f1(h_n);
							double f2_n = 0-f2(h, t_n);
							double f3_n = 0-f3(h, r_n, t);
							//double f1_n = f1(h_n);
							//double f2_n = f2(h, t_n);
							//double f3_n = f3(h, r_n, t);
							
							double sig_f1 = sigmoid(f1);
							double sig_f2 = sigmoid(f2);
							double sig_f3 = sigmoid(f3);
							double sig_f1_n = sigmoid(f1_n);
							double sig_f2_n = sigmoid(f2_n);
							double sig_f3_n = sigmoid(f3_n);
							
							double ln_sig_f1 = ln_sigmoid(f1);
							double ln_sig_f1_n = ln_sigmoid(f1_n);
							double ln_sig_f2 = ln_sigmoid(f2);
							double ln_sig_f2_n = ln_sigmoid(f2_n);
							double ln_sig_f3 = ln_sigmoid(f3);
							double ln_sig_f3_n = ln_sigmoid(f3_n);
							
							loss += ln_sig_f1 + ln_sig_f1_n + ln_sig_f2 + ln_sig_f2_n + ln_sig_f3 + ln_sig_f3_n;
//							loss += sig_f1 / sig_f1_n * sig_f2 / sig_f2_n * sig_f3 / sig_f3_n;
//							logger.info(loss+" "+ln_sig_f1+" "+ln_sig_f1_n+" "+ln_sig_f2+" "+ln_sig_f2_n+" "+ln_sig_f3+" "+ln_sig_f3_n);
							
							// to ensure that entityUpdateMap and relationUpdateMap contains h, r, t, h_n, r_n, t_n
							if(!entityUpdateMap.containsKey(h))
								entityUpdateMap.put(h, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
							if(!relationUpdateMap.containsKey(r>0?r:-r))
								relationUpdateMap.put(r>0?r:-r, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
							if(!entityUpdateMap.containsKey(t))
								entityUpdateMap.put(t, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
							if(!entityUpdateMap.containsKey(h_n))
								entityUpdateMap.put(h_n, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
							if(!relationUpdateMap.containsKey(r_n>0?r_n:-r_n))
								relationUpdateMap.put(r_n>0?r_n:-r_n, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
							if(!entityUpdateMap.containsKey(t_n))
								entityUpdateMap.put(t_n, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
							
							double delta = 0;
							// update h
							Set<Neighbor> neighbors = dataOperator.neighborMap.get(h);
							if(neighbors!=null){
								Iterator<Neighbor> itr = neighbors.iterator();
								while(itr.hasNext()){
									Neighbor neighbor = itr.next();
									int r_tmp = neighbor.r;
									int t_tmp = neighbor.t;
									if(!relationUpdateMap.containsKey(r_tmp>0?r_tmp:-r_tmp))
										relationUpdateMap.put(r_tmp>0?r_tmp:-r_tmp, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
									if(!entityUpdateMap.containsKey(t_tmp))
										entityUpdateMap.put(t_tmp, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
									delta = rate/batchSize*(1-sig_f1)/neighbors.size();
									for(int i=0; i<n; i++){
										if(r_tmp>0){
											if(entityVec[h][i] + relationVec[r_tmp][i] - entityVec[t_tmp][i] > 0){
												entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(-1));
												relationUpdateMap.get(r_tmp).set(i, relationUpdateMap.get(r_tmp).get(i) + delta*(-1));
												entityUpdateMap.get(t_tmp).set(i, entityUpdateMap.get(t_tmp).get(i) + delta*(1));
											}
											else{
												entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(1));
												relationUpdateMap.get(r_tmp).set(i, relationUpdateMap.get(r_tmp).get(i) + delta*(1));
												entityUpdateMap.get(t_tmp).set(i, entityUpdateMap.get(t_tmp).get(i) + delta*(-1));
											}
										}
										else{// r_tmp<=0
											if(entityVec[t_tmp][i] + relationVec[-r_tmp][i] - entityVec[h][i] > 0){
												entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(1));
												relationUpdateMap.get(-r_tmp).set(i, relationUpdateMap.get(-r_tmp).get(i) + delta*(-1));
												entityUpdateMap.get(t_tmp).set(i, entityUpdateMap.get(t_tmp).get(i) + delta*(-1));
											}
											else{
												entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(-1));
												relationUpdateMap.get(-r_tmp).set(i, relationUpdateMap.get(-r_tmp).get(i) + delta*(1));
												entityUpdateMap.get(t_tmp).set(i, entityUpdateMap.get(t_tmp).get(i) + delta*(1));
											}
										}
										
									}
								}
							}
							
							// update h_n
							neighbors = dataOperator.neighborMap.get(h_n);
							if(neighbors!=null){
								Iterator<Neighbor> itr = neighbors.iterator();
								while(itr.hasNext()){
									Neighbor neighbor = itr.next();
									int r_tmp = neighbor.r;
									int t_tmp = neighbor.t;
									if(!entityUpdateMap.containsKey(h_n))
										entityUpdateMap.put(h_n, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
									if(!relationUpdateMap.containsKey(r_tmp>0?r_tmp:-r_tmp))
										relationUpdateMap.put(r_tmp>0?r_tmp:-r_tmp, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
									if(!entityUpdateMap.containsKey(t_tmp))
										entityUpdateMap.put(t_tmp, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
									delta = rate/batchSize*(1-sig_f1_n)/neighbors.size();
									for(int i=0; i<n; i++){
										if(r_tmp>0){
											if(entityVec[h_n][i] + relationVec[r_tmp][i] - entityVec[t_tmp][i] > 0){
												entityUpdateMap.get(h_n).set(i, entityUpdateMap.get(h_n).get(i) + delta*(1));
												relationUpdateMap.get(r_tmp).set(i, relationUpdateMap.get(r_tmp).get(i) + delta*(1));
												entityUpdateMap.get(t_tmp).set(i, entityUpdateMap.get(t_tmp).get(i) + delta*(-1));
											}
											else{
												entityUpdateMap.get(h_n).set(i, entityUpdateMap.get(h_n).get(i) + delta*(-1));
												relationUpdateMap.get(r_tmp).set(i, relationUpdateMap.get(r_tmp).get(i) + delta*(-1));
												entityUpdateMap.get(t_tmp).set(i, entityUpdateMap.get(t_tmp).get(i) + delta*(1));
											}
										}
										else{// r_tmp<=0
											if(entityVec[h_n][i] - relationVec[-r_tmp][i] - entityVec[t_tmp][i] > 0){
												entityUpdateMap.get(h_n).set(i, entityUpdateMap.get(h_n).get(i) + delta*(1));
												relationUpdateMap.get(-r_tmp).set(i, relationUpdateMap.get(-r_tmp).get(i) + delta*(-1));
												entityUpdateMap.get(t_tmp).set(i, entityUpdateMap.get(t_tmp).get(i) + delta*(-1));
											}
											else{
												entityUpdateMap.get(h_n).set(i, entityUpdateMap.get(h_n).get(i) + delta*(-1));
												relationUpdateMap.get(-r_tmp).set(i, relationUpdateMap.get(-r_tmp).get(i) + delta*(1));
												entityUpdateMap.get(t_tmp).set(i, entityUpdateMap.get(t_tmp).get(i) + delta*(1));
											}
										}
									}
								}
							}

							// update t
							List<RelationPath> paths = dataOperator.pathMap.get(h+" "+t);
							if(paths!=null){
								Iterator<RelationPath> itr = paths.iterator();
								while(itr.hasNext()){
									RelationPath path = itr.next();
									delta = rate/batchSize*(1-sig_f2)/paths.size();
									for(int i=0; i<n; i++){
										double sum = entityVec[h][i] - entityVec[t][i];
										for(int j=0; j<path.size(); j++){
											int r_tmp = path.get(j);
											if(r_tmp>0){
												sum += relationVec[r_tmp][i];
											}
											else{
												sum -= relationVec[-r_tmp][i];
											}
										}
										// get dalta
										if(sum > 0){
											entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(-1));
											entityUpdateMap.get(t).set(i, entityUpdateMap.get(t).get(i) + delta*(1));
											for(int r_tmp : path.path){
												if(r_tmp>0){
													if(!relationUpdateMap.containsKey(r_tmp))
														relationUpdateMap.put(r_tmp, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
													relationUpdateMap.get(r_tmp).set(i, relationUpdateMap.get(r_tmp).get(i) + delta*(-1));
												}
												else{
													if(!relationUpdateMap.containsKey(-r_tmp))
														relationUpdateMap.put(-r_tmp, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
													relationUpdateMap.get(-r_tmp).set(i, relationUpdateMap.get(-r_tmp).get(i) + delta*(1));
												}
											}
										}
										else{
											entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(1));
											entityUpdateMap.get(t).set(i, entityUpdateMap.get(t).get(i) + delta*(-1));
											for(int r_tmp : path.path){
												if(r_tmp>0){
													if(!relationUpdateMap.containsKey(r_tmp))
														relationUpdateMap.put(r_tmp, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
													relationUpdateMap.get(r_tmp).set(i, relationUpdateMap.get(r_tmp).get(i) + delta*(1));
												}
												else{
													if(!relationUpdateMap.containsKey(-r_tmp))
														relationUpdateMap.put(-r_tmp, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
													relationUpdateMap.get(-r_tmp).set(i, relationUpdateMap.get(-r_tmp).get(i) + delta*(-1));
												}
											}
										}
									}
								}
							}
							
							// update t_n
							paths = dataOperator.pathMap.get(h+" "+t_n);
							if(paths!=null){
								Iterator<RelationPath> itr = paths.iterator();
								while(itr.hasNext()){
									RelationPath path = itr.next();
									delta = rate/batchSize*(1-sig_f2_n)/paths.size();
									for(int i=0; i<n; i++){
										double sum = entityVec[h][i] - entityVec[t_n][i];
										for(int j=0; j<path.size(); j++){
											int r_tmp = path.get(j);
											if(r_tmp>0){
												sum += relationVec[r_tmp][i];
											}
											else{
												sum -= relationVec[-r_tmp][i];
											}
										}
										// get dalta
										if(sum > 0){
											entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(1));
											entityUpdateMap.get(t_n).set(i, entityUpdateMap.get(t_n).get(i) + delta*(-1));
											for(int r_tmp : path.path){
												if(r_tmp>0){
													if(!relationUpdateMap.containsKey(r_tmp))
														relationUpdateMap.put(r_tmp, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
													relationUpdateMap.get(r_tmp).set(i, relationUpdateMap.get(r_tmp).get(i) + delta*(1));
												}
												else{
													if(!relationUpdateMap.containsKey(-r_tmp))
														relationUpdateMap.put(-r_tmp, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
													relationUpdateMap.get(-r_tmp).set(i, relationUpdateMap.get(-r_tmp).get(i) + delta*(-1));
												}
											}
										}
										else{
											entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(-1));
											entityUpdateMap.get(t_n).set(i, entityUpdateMap.get(t_n).get(i) + delta*(1));
											for(int r_tmp : path.path){
												if(r_tmp>0){
													if(!relationUpdateMap.containsKey(r_tmp))
														relationUpdateMap.put(r_tmp, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
													relationUpdateMap.get(r_tmp).set(i, relationUpdateMap.get(r_tmp).get(i) + delta*(-1));
												}
												else{
													if(!relationUpdateMap.containsKey(-r_tmp))
														relationUpdateMap.put(-r_tmp, new ArrayList<Double>(){{for(int i=0;i<n;i++) add(0.0);}});
													relationUpdateMap.get(-r_tmp).set(i, relationUpdateMap.get(-r_tmp).get(i) + delta*(1));
												}
											}
										}
									}
								}
							}
							
							// update r
							delta = rate/batchSize*(1-sig_f3);
							for(int i=0; i<n; i++){
								if(r>0){
									if(entityVec[h][i] + relationVec[r][i] - entityVec[t][i] > 0) {
										entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(-1));
										relationUpdateMap.get(r).set(i, relationUpdateMap.get(r).get(i) + delta*(-1));
										entityUpdateMap.get(t).set(i, entityUpdateMap.get(t).get(i) + delta*(1));
									}
									else{
										entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(1));
										relationUpdateMap.get(r).set(i, relationUpdateMap.get(r).get(i) + delta*(1));
										entityUpdateMap.get(t).set(i, entityUpdateMap.get(t).get(i) + delta*(-1));
									}
								}
								else{// r<=0
									if(entityVec[h][i] - relationVec[-r][i] - entityVec[t][i] > 0) {
										entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(-1));
										relationUpdateMap.get(-r).set(i, relationUpdateMap.get(-r).get(i) + delta*(1));
										entityUpdateMap.get(t).set(i, entityUpdateMap.get(t).get(i) + delta*(1));
									}
									else{
										entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(1));
										relationUpdateMap.get(-r).set(i, relationUpdateMap.get(-r).get(i) + delta*(-1));
										entityUpdateMap.get(t).set(i, entityUpdateMap.get(t).get(i) + delta*(-1));
									}
								}
							}
							
							// update r_n
							delta = rate/batchSize*(1-sig_f3_n);
							for(int i=0; i<n; i++){
								if(r_n>0){
									if(entityVec[h][i] + relationVec[r_n][i] - entityVec[t][i] > 0) {
										entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(1));
										relationUpdateMap.get(r_n).set(i, relationUpdateMap.get(r_n).get(i) + delta*(1));
										entityUpdateMap.get(t).set(i, entityUpdateMap.get(t).get(i) + delta*(-1));
									}
									else{
										entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(-1));
										relationUpdateMap.get(r_n).set(i, relationUpdateMap.get(r_n).get(i) + delta*(-1));
										entityUpdateMap.get(t).set(i, entityUpdateMap.get(t).get(i) + delta*(1));
									}
								}
								else{// r_n<=0
									if(entityVec[h][i] - relationVec[-r_n][i] - entityVec[t][i] > 0) {
										entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(1));
										relationUpdateMap.get(-r_n).set(i, relationUpdateMap.get(-r_n).get(i) + delta*(-1));
										entityUpdateMap.get(t).set(i, entityUpdateMap.get(t).get(i) + delta*(-1));
									}
									else{
										entityUpdateMap.get(h).set(i, entityUpdateMap.get(h).get(i) + delta*(-1));
										relationUpdateMap.get(-r_n).set(i, relationUpdateMap.get(-r_n).get(i) + delta*(1));
										entityUpdateMap.get(t).set(i, entityUpdateMap.get(t).get(i) + delta*(1));
									}
								}
							}
						}});
					
				}
				
				// wait for threads to finish
				fixedThreadPool.shutdown();
				try {
					while (!fixedThreadPool.awaitTermination(10, TimeUnit.SECONDS))
						;
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				// update temp entity and relation vectors and normalize them after iterate over the whole batch
				for(Entry<Integer, List<Double>> entry : entityUpdateMap.entrySet()){
					int updateEntity = entry.getKey();
					List<Double> updateList = entry.getValue();
					for(int j=0; j<n; j++)
						entityVec[updateEntity][j] += updateList.get(j);
					normalize(entityVec[updateEntity]);
				}
				for(Entry<Integer, List<Double>> entry : relationUpdateMap.entrySet()){
					int updateRelation = entry.getKey();
					List<Double> updateList = entry.getValue();
					for(int j=0; j<n; j++)
						relationVec[updateRelation][j] += updateList.get(j);
					normalize(relationVec[updateRelation]);
				}
				
//				entityVec = entityVec.clone();
//				relationVec = relationVec.clone();
			}

			logger.info("epoch:" + epoch + " loss:" + loss);

			// write vectors to files
			if(epoch>0.9*epochNum){ // only write the last 10% to file
				PrintWriter entityVecWriter = new PrintWriter(outDir+"/entity2vec."+n+"."+epoch, "UTF-8");
				PrintWriter relationVecWriter = new PrintWriter(outDir+"/relation2vec."+n+"."+epoch, "UTF-8");
				for (int i = 0; i < dataOperator.entityNum; i++) {
					for (int j = 0; j < n; j++)
						entityVecWriter.printf("%.6f\t", entityVec[i][j]);
					entityVecWriter.print("\n");
				}
				for (int i = 1; i < dataOperator.relationNum; i++) {
					for (int j = 0; j < n; j++)
						relationVecWriter.printf("%.6f\t", relationVec[i][j]);
					relationVecWriter.print("\n");
				}
				entityVecWriter.close();
				relationVecWriter.close();
			}
			
		}
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

//	void prepare() throws IOException {
//
//	}
	
	public TrainModel() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public TrainModel(String dataset, int n, double rate, int epochNum, int batchNum, int threadNum, String pathContextPath, String neighborContextPath, String entity2VecPath, String relation2VecPath, String outDir) throws IOException {
		super();
		this.dataset = dataset;
		this.n = n;
		this.rate = rate;
		this.epochNum = epochNum;
		this.batchNum = batchNum;
		this.threadNum = threadNum;
		this.neighborContextPath = neighborContextPath;
		this.pathContextPath = pathContextPath;
		this.entity2VecPath = entity2VecPath;
		this.relation2VecPath = relation2VecPath;
		this.outDir = outDir;
		
		if(this.entity2VecPath==null && this.relation2VecPath==null)
			this.init = true;
		else if(this.entity2VecPath!=null && this.relation2VecPath!=null)
			this.init = false;
		else{
			System.err.println("entity2VecPath and relation2VecPath should both be null or both not!");
			System.exit(1);
		}
		
		this.dataOperator = new DataOperator(this.dataset);
		this.dataOperator.prepare(this.pathContextPath, this.neighborContextPath);
		if(this.init)
			initVectors();
		else
			loadVectors(this.entity2VecPath, this.relation2VecPath); // load trained vectors from files
	}

	public static void main(String[] args) throws IOException {
		JsonParser parser = new JsonParser();
		JsonObject config = parser.parse(new FileReader("trainConfig.json")).getAsJsonObject();
		String dataset = config.get("dataset").getAsString();
		int n = config.get("n").getAsInt();
		double rate = config.get("rate").getAsDouble();
		int epochNum = config.get("epochNum").getAsInt();
		int batchNum = config.get("batchNum").getAsInt();
		int threadNum = config.get("threadNum").getAsInt();
		String pathContextPath = config.get("pathContextPath").getAsString();
		String neighborContextPath = config.get("neighborContextPath").getAsString();
		String entity2VecPath = config.get("entity2VecPath").isJsonNull()?null:config.get("entity2VecPath").getAsString();
		String relation2VecPath = config.get("relation2VecPath").isJsonNull()?null:config.get("relation2VecPath").getAsString();
		String outDir = config.get("outDir").getAsString();
		
		TrainModel trainModel = new TrainModel(dataset, n, rate, epochNum, batchNum, threadNum, neighborContextPath, pathContextPath, entity2VecPath, relation2VecPath, outDir);
		
		// args: 0-dataset 1-epochNum 2-dimensions 3-threadNum 4-pathMapPath 5-neighborMapPath 6-outDir (7-initEntityVec 8-initRelationVec)
		// java -jar trainModel.jar fb15k 1000 50 30 vec_fb15k_50_topPath_load vec_thu_fb15k_50/entity2vec.bern.999 vec_thu_fb15k_50/relation2vec.bern.999
//		int max_arg_num = 9;
//		if(!(args.length==max_arg_num-2) && !(args.length==max_arg_num))
//			return;
		
		// dimention, rate, epochs, batches
//		int n = 50;
//		double rate = 0.001;
//		int epochNum = Integer.valueOf(args[1]);
//		int batchNum = 100;
//		TrainModel trainModel = new TrainModel(n, rate, epochNum, batchNum);
		Logger logger = trainModel.logger;
		logger.info("size: " + trainModel.n);
		logger.info("learning rate: " + trainModel.rate);
		logger.info("epoch num: " + trainModel.epochNum);
		logger.info("batch num: " + trainModel.batchNum);

//		String dataset = args[0];
//		trainModel.n = Integer.valueOf(args[2]);
//		int threadNum = Integer.valueOf(args[3]);
//		String pathMap_path = args[4];
//		String neighborMap_path = args[5];
//		String outDir = args[6];
		
		File outPath = new File(outDir);
		if(outPath.exists() && outPath.isDirectory()){ // empty the directory if it exists and recreate
			logger.info("deleting dir " + outDir);
			trainModel.deleteDir(outDir);
			logger.info("making dir " + outDir);
			outPath.mkdir();
		}
		else{ // create the ourDir if it does not exists
			logger.info("making dir " + outDir);
			outPath.mkdir();
		}
		
//		if(args.length==max_arg_num) {
//			trainModel.init = false;
//			trainModel.entity2VecPath = args[7];
//			trainModel.relation2VecPath = args[8];
//		}
		
//		trainModel.prepare();
		trainModel.train();

		logger.info("END");
	}
	
	public void deleteDir(String path){ 
	    File f=new File(path); 
	    if(f.isDirectory()){// if current file is a directory, delete contents recursively
	        String[] list=f.list(); 
	        for(int i=0;i<list.length;i++){ 
	        	deleteDir(path+"//"+list[i]); 
	        } 
	    }        
	    f.delete(); 
	}
	
}
