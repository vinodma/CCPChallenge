
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
val fl_jeckle=sc.textFile("/user/vmangipudi/ccpdata/jeckle")
val fl_heckle=sc.textFile("/user/vmangipudi/ccpdata/heckle")
val fl = fl_jeckle.union(fl_heckle)
 val mp = scala.collection.mutable.Map[String,Double]()

val corrected = fl.map(x=>x.replace("craetedAt","createdAt").replace("created_at","createdAt").replace("user_agent","userAgent").replace("session_id","sessionID").replace("\"\"","\"").replace("item_id","itemId"))

val jn = sqlContext.jsonRDD(corrected)
val childaccounts = jn.filter("payload.subAction ='parentalControls'").select("user").distinct.map(_.getLong(0)).collect
val adultaccounts = jn.filter("payload.subAction ='updatePassword' or payload.subAction ='updatePaymentInfo'").select("user").distinct.map(_.getLong(0)).collect
val moviesrw = jn.select("payload.itemId").distinct.flatMap{rw=> val code=rw.getString(0);mp+=(code->0.0)}.collect.toMap
moviesrw.remove(null)
val allaccounts = jn.select("user").distinct.map(_.getLong(0))
val user_movie = jn.filter("type='Play'").select("user","payload.itemId").map(rw=>(rw.getLong(0),rw.getString(1)))

val user_movies = user_movie.reduceByKey((a,b)=>a+"," +b)

val u_m = user_movies.map{x=>  (x._1,x._2.split(",").distinct.mkString(","))}
val moviesrw_bc=sc.broadcast(moviesrw)
val childac_bc=sc.broadcast(childaccounts)
val adult_bc =sc.broadcast(adultaccounts)
val vect = u_m.map{um=> 
	val u=um._1;
	val m=um._2.split(",")
	val mp = collection.mutable.Map[String,Double]() ++= moviesrw_bc.value
	
	mp.remove(null)
	for(mv <- m){
	if(mp.contains(mv)){
	mp(mv)=1.0
	}
	}
	val childac=childac_bc.value
	val adultac=adult_bc.value
	var label = -1
	if(childac.contains(u)){
	 label = 1 
	 }
	 if(adultac.contains(u)) {
	 label = 0
	 } 
	(u,LabeledPoint(label,Vectors.dense(mp.values.toArray)))
	//mp.values.size
	}
	
val training = vect.map(a=>a._2).filter(v=>(v.label==1.0) ||  (v.label == 0.0))
val test = vect.map(a=>a._2).filter(v=>(v.label==-1.0)).map{case LabeledPoint(label, features)=>features}
val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)
val predictionAndLabels = training.map{ case LabeledPoint(label, features) =>val prediction = model.predict(features);(prediction, label)}
val metrics = new MulticlassMetrics(predictionAndLabels)
val testpreds=model.predict(test)
val user_label=vect.zip(testpreds)
