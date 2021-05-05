# Capstone Project

<h2 class="code-line" data-line-start=1 data-line-end=2 ><a id="And_Thats_the_Way_it_is_for_Now___Using_Network_Analysis_Event_Extraction_and_Named_Entity_Recognition_to_Track_Dynamic_News_Narratives_in_Written_Media_Coverage_1"></a>“And That’s the Way it is, for Now…”   Using Network Analysis, Event Extraction, and Named Entity Recognition to Track Dynamic News Narratives in Written Media Coverage</h2>
<h6 class="code-line" data-line-start=3 data-line-end=4 ><a id="As_newsworthy_events_unfold_their_narrative_coverage_evolves_as_well_It_is_difficult_to_keep_up_with_rapidly_changing_stories_Combining_Natural_Language_Processing_with_Network_Analysis_algorithms_and_techniques_and_a_highly_graphical_presentation_can_help_show_how_narratives_change__To_achieve_this_goal_first_network_analysis_algorithms_were_employed_to_aggregate_together_news_narrative_chains_Then_event_extraction_techniques_based_on_journalistic_structures_were_employed_to_determine_the_most_important_elements_being_reported_in_each_article_of_the_different_narratives_chains_Other_network_analysis_techniques_and_visualization_techniques_were_then_used_to_develop_a_highly_visual_method_for_clearly_differentiating_the_changing_narrative_threads_over_time__Additionally_Named_Entity_Recognition_was_used_to_detect_people_countries_locations_and_organizations_in_the_finalized_narrative_clusters__Finally_analysis_was_done_into_how_mentions_and_associations_of_these_entities_changed_over_the_course_of_each_narrative_chain_to_quantify_how_a_narrative_evolved_The_resulting_analysis_of_these_elements_graphically_shows_how_individual_news_narrative_chains_change_over_time_3"></a>As newsworthy events unfold, their narrative coverage evolves as well. It is difficult to keep up with rapidly changing stories. Combining Natural Language Processing with Network Analysis algorithms and techniques and a highly graphical presentation can help show how narratives change.  To achieve this goal, first, network analysis algorithms were employed to aggregate together news narrative chains. Then, event extraction techniques based on journalistic structures were employed to determine the most important elements being reported in each article of the different narratives chains. Other network analysis techniques and visualization techniques were then used to develop a highly visual method for clearly differentiating the changing narrative threads over time.  Additionally, Named Entity Recognition was used to detect people, countries, locations and organizations in the finalized narrative clusters.  Finally, analysis was done into how mentions and associations of these entities changed over the course of each narrative chain to quantify how a narrative evolved. The resulting analysis of these elements graphically shows how individual news narrative chains change over time.</h6>
<ul>


[![DOI](https://zenodo.org/badge/352833858.svg)](https://zenodo.org/badge/latestdoi/352833858)

The following are all of the files not uploaded due to their prohibitive size.  GitHub puts a file-size maximum of 50MB and a project-size maximum of 1GB.  Most of these files will not fit in the project, even if compressed.

```shell
paulinemckim@Paulines-MacBook-Pro-2 capstone % find . -type f -size +50M | grep -v ".git" | xargs -L1 ls -lah 
-rw-r--r--  1 paulinemckim  staff   111M Apr  6 14:19 ./Info_from_AWS_cluster_algo/Capstone/backup_after_10/Cache/Studykeywords.pickle
-rw-r--r--  1 paulinemckim  staff   150M Apr  6 14:23 ./Info_from_AWS_cluster_algo/Capstone/backup_after_10/WhooshIndex/MAIN_j13fqjtq7g9yrbvy.seg
-rw-r--r--  1 paulinemckim  staff   505M Apr  6 14:25 ./Info_from_AWS_cluster_algo/Capstone/backup_after_10/WhooshIndex/MAIN_pkua0srckuaocvoz.seg
-rw-r--r--  1 paulinemckim  staff   156M Apr  6 14:22 ./Info_from_AWS_cluster_algo/Capstone/backup_after_10/WhooshIndex/MAIN_c4d746ht3mj7d0xx.seg
-rw-r--r--  1 paulinemckim  staff   491M Apr  6 14:22 ./Info_from_AWS_cluster_algo/Capstone/backup_after_10/WhooshIndex/MAIN_bn9jdeji45kl8a4p.seg
-rw-r--r--  1 paulinemckim  staff   162M Apr  6 14:26 ./Info_from_AWS_cluster_algo/Capstone/backup_after_10/WhooshIndex/MAIN_ykcc2thb7qlj9l1d.seg
-rw-r--r--  1 paulinemckim  staff   111M Apr  6 14:32 ./Info_from_AWS_cluster_algo/Capstone/Cache/Studykeywords.pickle
-rw-r--r--  1 paulinemckim  staff   111M Apr  6 14:26 ./Info_from_AWS_cluster_algo/Capstone/backup_after_18/Cache/Studykeywords.pickle
-rw-r--r--  1 paulinemckim  staff   150M Apr  6 14:30 ./Info_from_AWS_cluster_algo/Capstone/backup_after_18/WhooshIndex/MAIN_j13fqjtq7g9yrbvy.seg
-rw-r--r--  1 paulinemckim  staff   505M Apr  6 14:31 ./Info_from_AWS_cluster_algo/Capstone/backup_after_18/WhooshIndex/MAIN_pkua0srckuaocvoz.seg
-rw-r--r--  1 paulinemckim  staff   156M Apr  6 14:29 ./Info_from_AWS_cluster_algo/Capstone/backup_after_18/WhooshIndex/MAIN_c4d746ht3mj7d0xx.seg
-rw-r--r--  1 paulinemckim  staff   491M Apr  6 14:29 ./Info_from_AWS_cluster_algo/Capstone/backup_after_18/WhooshIndex/MAIN_bn9jdeji45kl8a4p.seg
-rw-r--r--  1 paulinemckim  staff   162M Apr  6 14:32 ./Info_from_AWS_cluster_algo/Capstone/backup_after_18/WhooshIndex/MAIN_ykcc2thb7qlj9l1d.seg
-rw-r--r--  1 paulinemckim  staff    55M Apr  6 14:26 ./Info_from_AWS_cluster_algo/Capstone/backup_after_18/Output/hier_infomap_out_final/comm_plus_kw.pickle
-rw-r--r--  1 paulinemckim  staff   111M Apr  6 14:15 ./Info_from_AWS_cluster_algo/Capstone/backup_after_02/Cache/Studykeywords.pickle
-rw-r--r--  1 paulinemckim  staff   491M Apr  6 14:16 ./Info_from_AWS_cluster_algo/Capstone/backup_after_02/WhooshIndex/MAIN_bn9jdeji45kl8a4p.seg
-rw-r--r--  1 paulinemckim  staff   150M Apr  6 14:36 ./Info_from_AWS_cluster_algo/Capstone/WhooshIndex/MAIN_j13fqjtq7g9yrbvy.seg
-rw-r--r--  1 paulinemckim  staff   505M Apr  6 14:38 ./Info_from_AWS_cluster_algo/Capstone/WhooshIndex/MAIN_pkua0srckuaocvoz.seg
-rw-r--r--  1 paulinemckim  staff   156M Apr  6 14:36 ./Info_from_AWS_cluster_algo/Capstone/WhooshIndex/MAIN_c4d746ht3mj7d0xx.seg
-rw-r--r--  1 paulinemckim  staff   491M Apr  6 14:35 ./Info_from_AWS_cluster_algo/Capstone/WhooshIndex/MAIN_bn9jdeji45kl8a4p.seg
-rw-r--r--  1 paulinemckim  staff   162M Apr  6 14:39 ./Info_from_AWS_cluster_algo/Capstone/WhooshIndex/MAIN_ykcc2thb7qlj9l1d.seg
-rw-r--r--  1 paulinemckim  staff    68M Apr  6 14:33 ./Info_from_AWS_cluster_algo/Capstone/Output/hier_infomap_out_final/igraphPlusLabel.gml
-rw-r--r--  1 paulinemckim  staff    55M Apr  6 14:33 ./Info_from_AWS_cluster_algo/Capstone/Output/hier_infomap_out_final/comm_plus_kw.pickle
-rw-r--r--  1 paulinemckim  staff   468M Apr  6 14:14 ./Info_from_AWS_cluster_algo/Capstone/ArticleTexts/Guar_Small.CSV
-rw-r--r--  1 paulinemckim  staff   473M Apr  6 14:12 ./Info_from_AWS_cluster_algo/Capstone/ArticleTexts/Guar_Small-out.csv
-rw-r--r--  1 paulinemckim  staff   111M Apr  6 14:17 ./Info_from_AWS_cluster_algo/Capstone/backup_after_08/Cache/Studykeywords.pickle
-rw-r--r--  1 paulinemckim  staff   491M Apr  6 14:19 ./Info_from_AWS_cluster_algo/Capstone/backup_after_08/WhooshIndex/MAIN_bn9jdeji45kl8a4p.seg
-rw-r--r--  1 paulinemckim  staff   111M Apr  6 14:32 ./Info_from_AWS_cluster_algo/Capstone/backup_after_download/Cache/Studykeywords.pickle
-rw-r--r--  1 paulinemckim  staff   150M Apr  6 14:36 ./Info_from_AWS_cluster_algo/Capstone/backup_after_download/WhooshIndex/MAIN_j13fqjtq7g9yrbvy.seg
-rw-r--r--  1 paulinemckim  staff   505M Apr  6 14:38 ./Info_from_AWS_cluster_algo/Capstone/backup_after_download/WhooshIndex/MAIN_pkua0srckuaocvoz.seg
-rw-r--r--  1 paulinemckim  staff   156M Apr  6 14:36 ./Info_from_AWS_cluster_algo/Capstone/backup_after_download/WhooshIndex/MAIN_c4d746ht3mj7d0xx.seg
-rw-r--r--  1 paulinemckim  staff   491M Apr  6 14:35 ./Info_from_AWS_cluster_algo/Capstone/backup_after_download/WhooshIndex/MAIN_bn9jdeji45kl8a4p.seg
-rw-r--r--  1 paulinemckim  staff   162M Apr  6 14:39 ./Info_from_AWS_cluster_algo/Capstone/backup_after_download/WhooshIndex/MAIN_ykcc2thb7qlj9l1d.seg
-rw-r--r--  1 paulinemckim  staff    68M Apr  6 14:33 ./Info_from_AWS_cluster_algo/Capstone/backup_after_download/Output/hier_infomap_out_final/igraphPlusLabel.gml
-rw-r--r--  1 paulinemckim  staff    55M Apr  6 14:33 ./Info_from_AWS_cluster_algo/Capstone/backup_after_download/Output/hier_infomap_out_final/comm_plus_kw.pickle
-rw-r--r--  1 paulinemckim  staff   304M Apr 24 19:58 ./Data/Guardian/litte_set_5_to_9.csv
-rw-r--r--  1 paulinemckim  staff   378M Apr 30 09:45 ./Data/Guardian/stream-topped_2_to_4_second.csv
-rw-r--r--@ 1 paulinemckim  staff   468M Apr  9 14:22 ./Data/Guardian/Guar_Small.CSV
-rw-r--r--  1 paulinemckim  staff   306M Apr 30 09:34 ./Data/Guardian/stream-topped_5_to_9_second.csv
-rw-r--r--@ 1 paulinemckim  staff   2.7G Apr 11 17:10 ./Data/Guardian/ready_for_bert.csv
-rw-r--r--@ 1 paulinemckim  staff   582M Apr  4 22:00 ./Data/Guardian/Guar_Small_Test1.CSV
-rw-r--r--@ 1 paulinemckim  staff   191M Apr 25 10:57 ./Data/Guardian/stream-topped-1.csv
-rw-r--r--@ 1 paulinemckim  staff   102M Apr 20 21:45 ./Data/Guardian/stream-topped-2.csv
-rw-r--r--  1 paulinemckim  staff   605M Apr 24 19:57 ./Data/Guardian/litte_set_2_to_4.csv
-rw-r--r--  1 paulinemckim  staff   582M Apr  4 22:31 ./Data/Guardian/Guar_medium_api.CSV
-rw-r--r--@ 1 paulinemckim  staff   306M May  2 10:56 ./Data/Guardian/stream-topped_5_to_9_third.csv
-rw-r--r--  1 paulinemckim  staff   192M Apr 30 09:47 ./Data/Guardian/stream-topped_2_to_4_first.csv
-rw-r--r--@ 1 paulinemckim  staff   233M Apr 27 09:15 ./Data/Guardian/stream-topped_5_to_9_first.csv
```

