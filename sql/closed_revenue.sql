-- for each distinct value of stage in   expanded-nebula-754.SFDC.opportunities_as   sum amount, sum forecasted_close_value, and subtract the sum of amount from the sum of forecasted_close_value order by the count of each distinct stage values from high to low
SELECT
  stage,
  SUM(amount) AS sum_amount,
  SUM(forecasted_close_value) AS sum_forecasted_close_value,
  SUM(amount) - SUM(forecasted_close_value) AS difference_forecasted_minus_amount,
  COUNT(stage) AS count_of_stage_values
FROM
  `expanded-nebula-754`.`SFDC`.`opportunities_as`
GROUP BY
  stage
ORDER BY
  COUNT(stage) DESC;