import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import com.codingame.gameengine.runner.MultiplayerGameRunner;
import com.codingame.gameengine.runner.simulate.GameResult;
import com.google.common.io.Files;

public class Fall2023Main {
    public static void main(String[] args) throws IOException, InterruptedException {
        MultiplayerGameRunner gameRunner = new MultiplayerGameRunner();

        String agent1 = args[0];
        String agent2 = args[1];
        int level = Integer.parseInt(args[2]);

        gameRunner.addAgent("python " + agent1, "Agent_1");
        gameRunner.addAgent("python " + agent2, "Agent_2");
        gameRunner.setLeagueLevel(level);
        gameRunner.simulate();
    }

    private static String compile(String botFile) throws IOException, InterruptedException {

        File outFolder = Files.createTempDir();

        System.out.println("Compiling Boss.java... " + botFile);
        Process compileProcess = Runtime.getRuntime()
                .exec(new String[] { "bash", "-c", "javac " + botFile + " -d " + outFolder.getAbsolutePath() });
        compileProcess.waitFor();
        return "java -cp " + outFolder + " Player";
    }

    private static String[] compileTS(String botFile) throws IOException, InterruptedException {

        System.out.println("Compiling ... " + botFile);

        Process compileProcess = Runtime.getRuntime().exec(
                new String[] { "bash", "-c", "npx tsc --target ES2018 --inlineSourceMap --types ./typescript/readline/ "
                        + botFile + " --outFile /tmp/Boss.js" });
        compileProcess.waitFor();

        return new String[] { "bash", "-c", "node -r ./typescript/polyfill.js /tmp/Boss.js" };
    }
}
