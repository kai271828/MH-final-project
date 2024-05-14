package com.codingame.game;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.google.inject.Singleton;

@Singleton
public class GameSummaryManager {
    private List<String> lines;
    private Map<String, List<String>> playersErrors;
    private Map<String, List<String>> playersSummary;

    public GameSummaryManager() {
        this.lines = new ArrayList<>();
        this.playersErrors = new HashMap<>();
        this.playersSummary = new HashMap<>();
    }

    public String getSummary() {
        return toString();
    }

    public void clear() {
        this.lines.clear();
        this.playersErrors.clear();
        this.playersSummary.clear();
    }

    @Override
    public String toString() {
        return formatErrors() + "\n" + formatSummary() + "\n" + lines.stream().collect(Collectors.joining("\n"));
    }

    public void addError(Player player, String error) {
        String key = player.getNicknameToken();
        if (!playersErrors.containsKey(key)) {
            playersErrors.put(key, new ArrayList<String>());
        }
        playersErrors.get(key).add(error);
    }

    public void addPlayerSummary(String key, String summary) {
        if (!playersSummary.containsKey(key)) {
            playersSummary.put(key, new ArrayList<String>());
        }
        playersSummary.get(key).add(summary);
    }

    private String formatErrors() {
        return playersErrors.entrySet().stream().map(errorMap -> {
            List<String> errors = errorMap.getValue();
            String additionnalErrorsMessage = errors.size() > 1 ? " + " + (errors.size() - 1) + " other error" + (errors.size() > 2 ? "s" : "") : "";
            return errorMap.getKey() + ": " + errors.get(0) + additionnalErrorsMessage;
        }).collect(Collectors.joining("\n"));
    }

    public String formatSummary() {
        return playersSummary.entrySet().stream().flatMap(summaryMap -> {
            return summaryMap.getValue().stream();
        }).collect(Collectors.joining("\n"));
    }

}