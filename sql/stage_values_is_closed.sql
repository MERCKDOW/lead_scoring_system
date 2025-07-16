-- find the number of distinct values for stage in  expanded-nebula-754.SFDC.opportunities_as and count the number of occurrences for each distinct value. For each distinct value of stage count the number of values in is_closed that are true and count the number that are false. For each distinct value of stage count the number of values in is_won that are true and the number of values in is_won that are false
SELECT
  stage,
  COUNT(stage) AS count_of_occurrences,
  COUNTIF(is_closed IS TRUE) AS count_is_closed_true,
  COUNTIF(is_closed IS FALSE) AS count_is_closed_false,
  COUNTIF(is_won IS TRUE) AS count_is_won_true,
  COUNTIF(is_won IS FALSE) AS count_is_won_false
FROM
  `expanded-nebula-754`.`SFDC`.`opportunities_as`
GROUP BY
  stage
ORDER BY
  count_of_occurrences DESC;