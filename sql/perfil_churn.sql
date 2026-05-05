SELECT 
    active_member,
    AVG(churn) as churn_rate
FROM fact_customer_behavior
GROUP BY active_member;