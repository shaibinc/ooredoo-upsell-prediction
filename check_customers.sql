SELECT COUNT(*) as customer_count FROM customers;
GO

SELECT TOP 5 customer_id, age, gender, country, monthly_spend, upsell_target 
FROM customers 
ORDER BY customer_id DESC;
GO

SELECT country, COUNT(*) as count 
FROM customers 
GROUP BY country 
ORDER BY count DESC;
GO