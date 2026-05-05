SELECT 
    country_name,
    ROUND(AVG(churn) * 100, 2) AS churn_rate
FROM fact_customer_behavior f
JOIN dim_country c ON f.country_id = c.country_id
GROUP BY country_name;