-- count the number of row in expanded-nebula-754.SFDC.opportunities_as where is_won is true and count the number where is_won is false and count the number where is_won is niether
SELECT
  COUNTIF(t0.is_won IS TRUE) AS count_is_won_true,
  COUNTIF(t0.is_won IS FALSE) AS count_is_won_false,
  COUNTIF(t0.is_won IS NULL) AS count_is_won_neither
FROM
  `expanded-nebula-754`.`SFDC`.`opportunities_as` AS t0;