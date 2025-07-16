-- find the number of distinct values for stage in  expanded-nebula-754.SFDC.opportunities_as and count the number of occurrences for each distinct value
SELECT
  stage,
  COUNT(*) AS count_of_occurrences
FROM
  `expanded-nebula-754`.`SFDC`.`opportunities_as`
GROUP BY
  stage
ORDER BY
  count_of_occurrences DESC;